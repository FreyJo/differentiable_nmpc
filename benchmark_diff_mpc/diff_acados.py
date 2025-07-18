from typing import Optional
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosOcpBatchSolver

import casadi as ca
import numpy as np

import scipy.linalg

from timeit import default_timer as timer

from problems import ControlBoundedLqrProblem

def casadi_flatten(x: ca.SX) -> ca.SX:
    s = x.shape
    return x.reshape((s[0]*s[1], 1))

def solve_using_acados(problem: ControlBoundedLqrProblem, x0_vals,
                       seed: Optional[np.ndarray]=None, batched=False,
                       verify_kkt_residuals=False, vebose_solver_creation=False,
                       num_threads=None):

    if verify_kkt_residuals and batched:
        raise NotImplementedError("verify_kkt_residuals not implemented for batched solve.")
    if batched and num_threads is None:
        raise Exception("num_threads should be specified when using batched solve.")

    nx, nu = problem.nx, problem.nu
    nxu = nx + nu
    N_horizon = problem.N_horizon

    ocp = AcadosOcp()

    # model & dynamics
    model: AcadosModel = ocp.model
    model.x = ca.SX.sym('x', nx)
    model.u = ca.SX.sym('u', nu)
    model.name = 'linear_mpc'

    A_mat = ca.SX.sym('A', nx, nx)
    B_mat = ca.SX.sym('B', nx, nu)
    b = ca.SX.sym('b', nx)
    model.p_global = ca.vertcat(*[casadi_flatten(x) for x in [A_mat, B_mat, b]])
    ocp.p_global_values = np.concatenate((problem.dynamics.A.flatten(order='F'), problem.dynamics.B.flatten(order='F'), problem.dynamics.b))

    ocp.solver_options.integrator_type = "DISCRETE"
    model.disc_dyn_expr = A_mat @ model.x + B_mat @ model.u + b

    # define cost
    ocp.cost.cost_type = 'EXTERNAL'  # NOTE: if no sensitivities are needed using LINEAR_LS would be even more efficient.
    ocp.cost.cost_type_e = 'EXTERNAL'
    H_mat = ca.SX.sym('H', nxu, nxu)
    xu = ca.vertcat(model.x, model.u)
    ocp.model.cost_expr_ext_cost = ca.mtimes([xu.T, H_mat, xu])
    ocp.model.cost_expr_ext_cost_e = ca.mtimes([model.x.T, H_mat[:nx, :nx], model.x])
    ocp.model.p_global = ca.vertcat(*[casadi_flatten(x) for x in [ocp.model.p_global, H_mat]])
    H_mat_val = scipy.linalg.block_diag(problem.cost.Q, problem.cost.R)
    ocp.p_global_values = np.concatenate((ocp.p_global_values, H_mat_val.flatten(order='F')))

    # define constraints
    ocp.constraints.lbu = problem.control_bounds.u_lower
    ocp.constraints.ubu = problem.control_bounds.u_upper
    ocp.constraints.idxbu = np.arange(nu)

    # intial state dummy
    ocp.constraints.x0 = x0_vals[0,:]

    # solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_cond_N = int(N_horizon/4)

    if seed is not None:
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.with_solution_sens_wrt_params = True

    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.with_batch_functionality = True
    ocp.solver_options.tol = 1e-7
    if isinstance(problem, ControlBoundedLqrProblem):
        ocp.solver_options.nlp_solver_max_iter = 1  # SQP converges in 1 iteration for LQR

    du_dp_adj_batch = None
    n_batch = x0_vals.shape[0]
    if not batched:
        solver = AcadosOcpSolver(ocp, verbose=vebose_solver_creation, generate=True, build=True)

        time_start = timer()
        x_batch_sol = np.zeros((n_batch, N_horizon+1, nx))
        u_batch_sol = np.zeros((n_batch, N_horizon+1, nu))
        for i in range(n_batch):
            # set initial state
            solver.set(0, 'lbx', x0_vals[i,:])
            solver.set(0, 'ubx', x0_vals[i,:])
            # NOTE: Not necessary, but for fairness of comparison we are setting the parameters again
            solver.set_p_global_and_precompute_dependencies(ocp.p_global_values)
            # solve
            solver.solve()
            # get solution
            x_sol = solver.get_flat('x').reshape((N_horizon+1, nx))
            u_sol = solver.get_flat('u').reshape((N_horizon, nu))
            x_batch_sol[i,:,:] = x_sol
            u_batch_sol[i,:-1,:] = u_sol

            if verify_kkt_residuals:
                residuals = solver.get_residuals(recompute=True)
                if np.any(residuals > ocp.solver_options.tol):
                    raise Exception(f"KKT residuals are not zero: {residuals}")

        timing = timer() - time_start

    else:
        solver = AcadosOcpBatchSolver(ocp, N_batch_max=n_batch, verbose=vebose_solver_creation, num_threads_in_batch_solve=num_threads)

        time_start = timer()
        x_batch_sol = np.zeros((n_batch, N_horizon+1, nx))
        u_batch_sol = np.zeros((n_batch, N_horizon+1, nu))

        if seed is not None:
            seed_u_batch = np.tile(seed, (n_batch, 1, 1))
            seed_u_batch = seed_u_batch.transpose((0, 2, 1))
            n_seed = 1
            du_dp_adj_batch = np.zeros((n_seed, ocp.dims.np_global))

        # set initial state
        for j in range(n_batch):
            solver.ocp_solvers[j].set(0, 'lbx', x0_vals[j,:])
            solver.ocp_solvers[j].set(0, 'ubx', x0_vals[j,:])

        # set parameters
        # NOTE: Not necessary, but for fairness of comparison we are setting the parameters again
        for j in range(n_batch):
            solver.ocp_solvers[j].set_p_global_and_precompute_dependencies(ocp.p_global_values)

        # solve
        solver.solve()
        # get solution
        x_batch_sol = solver.get_flat('x').reshape((n_batch, N_horizon+1, nx))
        u_batch_sol[:, :-1, :] = solver.get_flat('u').reshape((n_batch, N_horizon, nu))

        if seed is not None:
            solver.setup_qp_matrices_and_factorize()
            p_sens_ = solver.eval_adjoint_solution_sensitivity(seed_x=None, seed_u=[(0, seed_u_batch)], sanity_checks=False, )
            du_dp_adj_batch += np.sum(p_sens_, axis=0)

        timing = timer() - time_start
        if seed is not None:
            du_dp_adj_batch = du_dp_adj_batch.flatten()

    # reshape to match mpc.pytorch
    x_batch_sol = x_batch_sol.transpose((1,0,2))
    u_batch_sol = u_batch_sol.transpose((1,0,2))

    return x_batch_sol, u_batch_sol, timing, du_dp_adj_batch

