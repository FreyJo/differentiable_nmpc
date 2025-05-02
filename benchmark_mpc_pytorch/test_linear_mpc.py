import torch
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import cvxpy as cp
import scipy

from timeit import default_timer as timer

from torch.autograd import Variable

from diff_acados import solve_using_acados

from linear_mpc import define_bounded_lqr_test_problem, create_batched_problem_data, TOL, solve_using_mpc_pytorch


def bounded_mpytorch_cvxpy(C, c, A, B, b, x0, N_horizon, nx, nu, u_lower, u_upper):
    """Solve min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = A_t x_t + B_t u_t + b_t
                             x_0 = x0
                             u_lower <= u <= u_upper
    """
    tau = cp.Variable((nx+nu, N_horizon))
    assert (u_lower is None) == (u_upper is None)

    objs = []
    x_0 = tau[:nx,0]
    cons = [x_0 == x0]
    for t in range(N_horizon):
        xt = tau[:nx,t]
        ut = tau[nx:,t]
        objs.append(0.5*cp.quad_form(tau[:,t], C[t]) +
                    cp.sum(cp.multiply(c[t], tau[:,t])))
        if u_lower is not None:
            cons += [u_lower[t] <= ut, ut <= u_upper[t]]
        if t+1 < N_horizon:
            xtp1 = tau[:nx, t+1]
            cons.append(xtp1 == A[t] @ xt + B[t] @ ut + b[t])
    prob = cp.Problem(cp.Minimize(sum(objs)), cons)
    # prob.solve(solver=cp.SCS, verbose=True)
    prob.solve(solver='CLARABEL', tol_feas=1e-1*TOL, tol_infeas_abs=1e-1*TOL, tol_gap_abs=1e-1*TOL)
    assert 'optimal' in prob.status
    return np.array(tau.value).T, np.array([obj_t.value for obj_t in objs])


def solve_using_cvxpy(H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch):

    N_horizon, n_batch, nx, nu = B_batch.shape
    x_cp = np.zeros((N_horizon, n_batch, nx))
    u_cp = np.zeros((N_horizon, n_batch, nu))
    time_start = timer()
    for i in range(n_batch):
        tau, obj = bounded_mpytorch_cvxpy(
            H_batch[:,i], c_batch[:,i], A_batch[:,i], B_batch[:, i], b_batch[:,i], x0[i], N_horizon, nx, nu,
            u_lower_batch[:,i], u_upper_batch[:,i]
        )
        x_cp[:, i, :] = tau[:,:nx]
        u_cp[:, i, :] = tau[:,nx:]
    timing = timer() - time_start

    return x_cp, u_cp, timing


def control_bounded_lqr_test(umax: float = 1.0):
    npr.seed(42)
    problem = define_bounded_lqr_test_problem(umax=umax)

    # get dimensions
    N_horizon = problem.N_horizon
    nx, nu = problem.nx, problem.nu

    # dimensions
    n_batch = 32
    x0 = npr.randn(n_batch, nx)

    # create batched problem data
    H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

    # solve using cvxpy
    x_cp, u_cp, timing_cp = solve_using_cvxpy(H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch)

    # solve with acados (batched)
    test_tol_acados = 1e2 * TOL
    num_threads = 4
    x_ac, u_ac, timing_ac, _ = solve_using_acados(problem, x0, batched=True, num_threads=num_threads)

    diff_cp_ac = max(np.max(np.abs([x_ac - x_cp])), np.max(np.abs([u_ac - u_cp])))
    print(f"difference acados vs cvxpy: {diff_cp_ac:.2e}")
    npt.assert_allclose(x_ac, x_cp, atol=test_tol_acados)
    npt.assert_allclose(u_ac, u_cp, atol=test_tol_acados)

    # solve with acados (sequential) + verify convergence of KKT residuals
    x_ac_seq, u_ac_seq, timing_ac_seq, _ = solve_using_acados(problem, x0, batched=False, verify_kkt_residuals=True)
    diff_cp_ac = max(np.max(np.abs([x_ac_seq - x_cp])), np.max(np.abs([u_ac_seq - u_cp])))
    print(f"difference acados sequential vs cvxpy: {diff_cp_ac:.2e}")
    npt.assert_allclose(x_ac_seq, x_cp, atol=test_tol_acados)
    npt.assert_allclose(u_ac_seq, u_cp, atol=test_tol_acados)
    print("\nacados and cvxpy solutions match as expected.\n")

    # convert to torch
    TEST_TOL = 1e3 * TOL
    H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch = [
        Variable(torch.Tensor(x).double()) if x is not None else None
        for x in [H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch]
    ]

    # solve using MPC class of mpc.pytorch
    x_mpytorch, u_mpytorch, timing_mpytorch, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch)
    x_mpytorch = x_mpytorch.detach().numpy()
    u_mpytorch = u_mpytorch.detach().numpy()

    print(f"Timings: {timing_ac=}, {timing_mpytorch=}")

    diff_cp_mpytorch = max(np.max(np.abs([x_mpytorch - x_cp])), np.max(np.abs([u_mpytorch - u_cp])))
    num_elements_exceeding_tol = np.sum(np.abs(x_mpytorch - x_cp) > TEST_TOL) + np.sum(np.abs(u_mpytorch - u_cp) > TEST_TOL)
    total_elements = np.prod(x_cp.shape) + np.prod(u_cp.shape)
    percentage_exceeding_tol = (num_elements_exceeding_tol / total_elements) * 100
    print(f"difference mpc.pytorch vs cvxpy: {diff_cp_mpytorch:.2e}")
    print(f"Percentage of elements with difference > TEST_TOL: {percentage_exceeding_tol:.2f}%")

    if diff_cp_mpytorch < TEST_TOL:
        print("\nmpc.pytorch and cvxpy solutions match as expected, for problems without active constraints.\n")
    else:
        print("\nmpc.pytorch DID NOT CONVERGE to correct solution.")
        if umax < 1e3:
            print("This is expected, as there are active constraints.")
        else:
            raise Exception("mpc.pytorch should converge for this problem.")



def control_bounded_lqr_sens_test(umax: float = 1.0):
    npr.seed(42)
    problem = define_bounded_lqr_test_problem(umax=umax)

    # get dimensions
    N_horizon = problem.N_horizon
    nx, nu = problem.nx, problem.nu

    # define seed
    seed = np.ones((nu,))

    # dimensions
    n_batch = 8
    x0 = npr.randn(n_batch, nx)

    # solve with acados (batched)
    test_tol = 1e2 * TOL
    num_threads = 4
    x_ac, u_ac, timing_ac, adj_sens_ac = solve_using_acados(problem, x0, batched=True, num_threads=num_threads, seed=seed)

    # create batched problem data
    H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

    # convert to torch
    H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch = [
        Variable(torch.Tensor(x).double()) if x is not None else None
        for x in [H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch]
    ]

    # convert to torch
    H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch = [
        Variable(torch.Tensor(x).double()) if x is not None else None
        for x in [H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch]
    ]
    # NOTE: compute gradients w.r.t. A, B, b, H
    c_batch = torch.tensor(np.concatenate([problem.cost.q, problem.cost.r]), requires_grad=False)
    A_batch = torch.tensor(problem.dynamics.A, requires_grad=True)
    B_batch = torch.tensor(problem.dynamics.B, requires_grad=True)
    b_batch = torch.tensor(problem.dynamics.b, requires_grad=True)
    H_batch = torch.tensor(scipy.linalg.block_diag(problem.cost.Q, problem.cost.R), requires_grad=True)

    # solve using MPC class of mpc.pytorch
    x_mpytorch, u_mpytorch, timing_mpytorch, adj_sens_mpytorch = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, seed=seed)
    x_mpytorch = x_mpytorch.detach().numpy()
    u_mpytorch = u_mpytorch.detach().numpy()

    print(f"Timings: {timing_ac=}, {timing_mpytorch=}")
    npt.assert_allclose(adj_sens_ac, adj_sens_mpytorch, atol=test_tol)
    print(f"Adjoint sensitivities of acados and mpc.pytorch match!")



def main():
    control_bounded_lqr_test(umax=1e4)
    control_bounded_lqr_test(umax=1.0)
    control_bounded_lqr_sens_test(umax=1e4)

if __name__ == "__main__":
    main()
