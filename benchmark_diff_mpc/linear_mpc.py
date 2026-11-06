import torch
import os
import math
import numpy as np
import numpy.random as npr
import cvxpy as cp
import importlib
import scipy.linalg

from cvxpylayers.torch import CvxpyLayer
from cvxpygen import cpg
from mpc import mpc, util
from mpc.mpc import QuadCost
from mpc.dynamics import AffineDynamics

from problems import LinearDiscreteDynamics, QuadraticCost, ControlBounds, ControlBoundedLqrProblem
from diff_acados import solve_using_acados

from timeit import default_timer as timer

npr.seed(42)

def define_bounded_lqr_test_problem(nx=8, nu=4, umax=1.0, N=20, tol=1e-6):

    Q_mat = np.eye(nx)
    R_mat = np.eye(nu)
    quadratic_cost = QuadraticCost(Q_mat, R_mat, np.zeros(nx), np.zeros(nu))

    alpha = 0.1
    A_mat = np.eye(nx)+alpha*npr.randn(nx, nx)
    B_mat = npr.randn(nx, nu)
    b = npr.randn(nx)

    dynamics = LinearDiscreteDynamics(A_mat, B_mat, b)
    control_bounds = ControlBounds(-umax*np.ones((nu,)), umax*np.ones((nu,)))

    ocp = ControlBoundedLqrProblem(dynamics, quadratic_cost, control_bounds, N_horizon=N, tol=tol)
    return ocp


def solve_using_cvxpy(
    H_batch,
    A_batch,
    B_batch,
    b_batch,
    x0,
    u_lower_batch,
    u_upper_batch,
    N,
    tol,
    seed=None,
    device="cpu",
    codegen_suff="",
    n_exec=1,
):
    """Solve min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t
    s.t. x_{t+1} = A_t x_t + B_t u_t + b_t
         x_0 = x0
         u_lower <= u <= u_upper
    """
    if n_exec < 1:
        raise ValueError("n_exec must be at least 1.")

    best_timing = math.inf
    best_x_cp_tch = None
    best_u_cp_tch = None
    best_du_dp = None

    # NOTE: no save way to fully reset solver, thus it is not reused.

    for i_exec in range(n_exec):
        if len(B_batch.shape) == 4:
            N_horizon, n_batch, nx, nu = B_batch.shape
            u_lower = u_lower_batch[0, 0]
            u_upper = u_upper_batch[0, 0]
            H_single = H_batch[0, 0]
            A_single = A_batch[0, 0]
            B_single = B_batch[0, 0]
            b_single = b_batch[0, 0]
            # NOTE: All parameters except x0 should actually
            # be constant over the horizon and batch.
            for i in range(n_batch):
                for j in range(N_horizon):
                    assert np.all(u_lower == u_lower_batch[j, i])
                    assert np.all(u_upper == u_upper_batch[j, i])
                    assert np.all(H_batch[j, i] == H_single)
                    assert np.all(A_batch[j, i] == A_single)
                    assert np.all(B_batch[j, i] == B_single)
                    assert np.all(b_batch[j, i] == b_single)
        elif len(B_batch.shape) == 2:
            nx, nu = B_batch.shape
            N_horizon = N + 1
            n_batch = x0.shape[0]
            u_lower = u_lower_batch[0]
            u_upper = u_upper_batch[0]
            for i in range(N_horizon):
                assert np.all(u_lower == u_lower_batch[i])
                assert np.all(u_upper == u_upper_batch[i])
            H_single = H_batch
            A_single = A_batch
            B_single = B_batch
            b_single = b_batch
        else:
            raise ValueError(
                "Input should either be of shape (N_horizon, N_batch, ...) or of shape (...)."
            )

        (H_single, A_single, B_single, b_single, x0) = [
            torch.Tensor(x).double().detach().to(device)
            for x in [H_single, A_single, B_single, b_single, x0]
        ]

        if seed is not None:
            A_single.requires_grad = True
            B_single.requires_grad = True
            b_single.requires_grad = True
            H_single.requires_grad = True

        x_cp_tch = torch.zeros((N_horizon, n_batch, nx), dtype=torch.float64)
        u_cp_tch = torch.zeros((N_horizon, n_batch, nu), dtype=torch.float64)

        tau = cp.Variable((nx + nu, N_horizon), name="tau")
        A = cp.Parameter((nx, nx), name="A")
        B = cp.Parameter((nx, nu), name="B")
        b = cp.Parameter((nx), name="b")
        C_sqrt = cp.Parameter((nx + nu, nx + nu), name="C_sqrt")
        x_init = cp.Parameter((nx), name="x_init")
        assert (u_lower is None) == (u_upper is None)

        objs = []
        x_0 = tau[:nx, 0]
        cons = [x_0 == x_init]
        for t in range(N_horizon):
            xt = tau[:nx, t]
            ut = tau[nx:, t]
            objs.append(0.5 * cp.sum_squares(C_sqrt @ tau[:, t]))
            if u_lower is not None:
                cons += [
                    u_lower <= ut,
                    ut <= u_upper,
                ]
            if t + 1 < N_horizon:
                xtp1 = tau[:nx, t + 1]
                cons.append(xtp1 == A @ xt + B @ ut + b)
        prob = cp.Problem(cp.Minimize(sum(objs)), cons)

        assert prob.is_dcp()
        assert prob.is_dpp()
        solver_args = {
            "problem": prob,
            "warm_start": True,
            "eps_abs": tol,
            "eps_rel": tol,
            "max_iter": 100000,
        }

        codegen_suff += f"_{i_exec}"
        codegen_mod_name = "lmpc_convex_codegen" + codegen_suff
        cpg.generate_code(
            prob,
            code_dir=codegen_mod_name,
            solver="OSQP",
            gradient=True,
            wrapper=True,
            prefix=codegen_suff
        )
        module = importlib.import_module(f"{codegen_mod_name}.cpg_solver")
        forward = getattr(module, "forward")
        backward = getattr(module, "backward")
        # from lmpc_convex_codegen.cpg_solver import backward, forward  # noqa: F401

        layer = CvxpyLayer(
            prob,
            parameters=[A, B, b, C_sqrt, x_init],
            variables=[tau],
            custom_method=(forward, backward),
        ).to(device)

        C_sqrt_single = torch.linalg.cholesky(H_single, upper=True)

        all_params = ["A", "B", "b", "C_sqrt", "x_init"]
        only_x = ["x_init"]

        if seed is not None:
            for param in [A_single, B_single, b_single, H_single]:
                if param.grad is not None:
                    param.grad.zero_()

        x_cp_tch = torch.zeros((N_horizon, n_batch, nx), dtype=torch.float64, device=device)
        u_cp_tch = torch.zeros((N_horizon, n_batch, nu), dtype=torch.float64, device=device)

        time_start = timer()
        for i in range(n_batch):
            if i == 0:
                solver_args["updated_params"] = all_params
            else:
                solver_args["updated_params"] = only_x
            (tau,) = layer(
                A_single,
                B_single,
                b_single,
                C_sqrt_single,
                x0[i],
                solver_args=solver_args,
            )
            x_cp_tch[:, i, :] = tau[:nx, :].T
            u_cp_tch[:, i, :] = tau[nx:, :].T

        if seed is None:
            du_dp = None
        else:
            u0_star = u_cp_tch[0, :, :].sum(dim=0)
            u0_star.backward(torch.ones_like(u0_star))
            du_dp = np.concatenate(
                (
                    A_single.grad.cpu().numpy().flatten(order="F"),
                    B_single.grad.cpu().numpy().flatten(order="F"),
                    b_single.grad.cpu().numpy().flatten(),
                    H_single.grad.cpu().numpy().flatten(order="F"),
                )
            )

        timing = timer() - time_start
        if timing < best_timing:
            best_timing = timing
            best_x_cp_tch = x_cp_tch.detach().cpu().clone()
            best_u_cp_tch = u_cp_tch.detach().cpu().clone()
            best_du_dp = du_dp

    return (
        best_x_cp_tch.numpy(),
        best_u_cp_tch.numpy(),
        best_timing,
        best_du_dp,
    )

def solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, lbu, ubu, tol,
                            seed=None, device="cpu", n_exec=1):
    N_horizon += 1

    if len(B_batch.shape) == 4:
        H_single = H_batch[0, 0]
        A_single = A_batch[0, 0]
        B_single = B_batch[0, 0]
        b_single = b_batch[0, 0]
        c_single = c_batch[0, 0]
        # NOTE: All parameters except x0, ubu, lbu should actually
        # be constant over the horizon and batch.
        for i in range(n_batch):
            for j in range(N_horizon):
                assert np.all(H_batch[j, i] == H_single)
                assert np.all(A_batch[j, i] == A_single)
                assert np.all(B_batch[j, i] == B_single)
                assert np.all(b_batch[j, i] == b_single)
                assert np.all(c_batch[j, i] == c_single)
        H_batch = H_single
        c_batch = c_single
        A_batch = A_single
        B_batch = B_single
        b_batch = b_single
    elif len(B_batch.shape) == 2:
        pass
    else:
        raise ValueError(
            "Input should either be of shape (N_horizon, N_batch, ...) or of shape (...)."
        )

    # convert to torch
    H_batch, c_batch, A_batch, B_batch, b_batch, x0, lbu, ubu = [
        torch.Tensor(x).double().detach().to(device)
        for x in [H_batch, c_batch, A_batch, B_batch, b_batch, x0, lbu, ubu]
    ]

    if seed is not None:
        A_batch.requires_grad = True
        B_batch.requires_grad = True
        b_batch.requires_grad = True
        H_batch.requires_grad = True

    if n_exec < 1:
        raise ValueError("n_exec must be at least 1.")

    best_timing = math.inf
    best_x_mpytorch = None
    best_u_mpytorch = None
    best_du_dp_adj = None

    for _ in range(n_exec):
        dynamics = AffineDynamics(A_batch, B_batch, b_batch)
        quad_cost = QuadCost(H_batch, c_batch)
        mpc_pytorch_solver = mpc.MPC(
            nx, nu, N_horizon, lbu, ubu, None,
            lqr_iter=100,  # NOTE: defines max iterations for LQR solver, with the default of 10, it fails on many more problems.
            backprop=(seed is not None),
            exit_unconverged=False,
            eps=tol,
            n_batch=n_batch,
        )

        if seed is not None:
            for param in [A_batch, B_batch, b_batch, H_batch]:
                if param.grad is not None:
                    param.grad.zero_()

        time_start = timer()
        x_mpytorch_, u_mpytorch_, objs_mpytorch = mpc_pytorch_solver(x0, quad_cost, dynamics)
        x_mpytorch = util.get_data_maybe(x_mpytorch_)
        u_mpytorch = util.get_data_maybe(u_mpytorch_)

        if seed is None:
            du_dp_adj = None
        else:
            loss = torch.sum(u_mpytorch_[0, :, :])
            loss.backward()
            du_dp_adj = np.concatenate((A_batch.grad.cpu().numpy().flatten(order='F'),
                                        B_batch.grad.cpu().numpy().flatten(order='F'),
                                        b_batch.grad.cpu().numpy().flatten(),
                                        H_batch.grad.cpu().numpy().flatten(order='F'),
                                        ))

        timing = timer() - time_start
        if timing < best_timing:
            best_timing = timing
            best_x_mpytorch = x_mpytorch.detach().cpu().clone().numpy()
            best_u_mpytorch = u_mpytorch.detach().cpu().clone().numpy()
            best_du_dp_adj = du_dp_adj

    # if du_dp_adj is not None:
    #     du_dp_adj = du_dp_adj.cpu()

    return best_x_mpytorch, best_u_mpytorch, best_timing, best_du_dp_adj


def control_bounded_lqr_solve_and_adj_sens_timings(problem: ControlBoundedLqrProblem, x0, with_acados=True, with_mpc_pytorch = False, with_mpc_pytorch_cuda = False, with_cvxpy = False, with_cvxpy_cuda = False, num_threads=None, codegen_suff="", n_exec=1):
    if with_mpc_pytorch_cuda:
        assert with_mpc_pytorch
    if with_cvxpy_cuda:
        assert with_cvxpy_cuda

    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    # get dimensions
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon
    tol = problem.tol

    # define seed
    seed = np.ones((nu,))

    # dimensions
    n_batch = x0.shape[0]

    # solve with acados
    if with_acados:
        x_ac, u_ac, timing_ac, du_dp_adj_batch = solve_using_acados(problem, x0, seed=seed, batched=True, num_threads=num_threads, n_exec=n_exec)
        store_results(x_ac, u_ac, timing_ac, "acados", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, tol, sensitivity=True)

    if with_mpc_pytorch:
        # create batched problem data
        H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

        # solve using MPC class of mpc.pytorch
        x_mpytorch, u_mpytorch, timing_mpytorch, du_dp_adj_mpy = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, tol, seed=seed, n_exec=n_exec)
        store_results(x_mpytorch, u_mpytorch, timing_mpytorch, "mpc_pytorch", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, tol, sensitivity=True)
        # print(f"{u_mpytorch=}")

        if with_mpc_pytorch_cuda:
            # move to GPU
            device = "cuda"
            x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, du_dp_adj_mpy_cuda = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, tol, seed=seed, device=device, n_exec=n_exec)
            store_results(x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, "mpc_pytorch_cuda", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, tol, sensitivity=True)

    if with_cvxpy:
        # create batched problem data
        H_batch, _, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = (
            create_batched_problem_data(problem, n_batch)
        )

        # solve using cvxpygen + cvxpy layer
        x_cvxpy, u_cvxpy, timing_cvxpy, du_dp_adj_cvxpy = solve_using_cvxpy(
            H_batch,
            A_batch,
            B_batch,
            b_batch,
            x0,
            u_lower_batch,
            u_upper_batch,
            N_horizon,
            tol,
            seed=seed,
            codegen_suff=codegen_suff,
            n_exec=n_exec
        )
        store_results(
            x_cvxpy,
            u_cvxpy,
            timing_cvxpy,
            "cvxpy",
            problem.control_bounds.u_upper[0],
            nx,
            nu,
            n_batch,
            N_horizon,
            tol,
            sensitivity=True,
        )

        if with_cvxpy_cuda:
            # move to GPU
            device = "cuda"
            (
                x_cvxpy_cuda,
                u_cvxpy_cuda,
                timing_cvxpy_cuda,
                du_dp_adj_cvxpy_cuda,
            ) = solve_using_cvxpy(
                H_batch,
                A_batch,
                B_batch,
                b_batch,
                x0,
                u_lower_batch,
                u_upper_batch,
                N_horizon,
                tol,
                seed=seed,
                device=device,
                codegen_suff=codegen_suff,
                n_exec=n_exec
            )
            store_results(
                x_cvxpy_cuda,
                u_cvxpy_cuda,
                timing_cvxpy_cuda,
                "cvxpy_cuda",
                problem.control_bounds.u_upper[0],
                nx,
                nu,
                n_batch,
                N_horizon,
                tol,
                sensitivity=True,
            )

    print("Timings:")
    if with_acados:
        print(f"acados: {timing_ac=}")
    if with_mpc_pytorch:
        print(f"mpc-pytorch: {timing_mpytorch=}")
    if with_mpc_pytorch_cuda:
        print(f"mpc-pytorch-cuda: {timing_mpytorch_cuda=}")
    if with_acados and with_mpc_pytorch:
        # compare timings
        speedup = timing_mpytorch / timing_ac
        print(f"speedup: acados vs pytorch: {speedup:.3f}")
    if with_cvxpy:
        print(f"cvxpy (gen + layer): {timing_cvxpy=}")
    if with_cvxpy_cuda:
        print(f"cvxpy_cuda (gen + layer): {timing_cvxpy_cuda=}")
    if with_acados and with_cvxpy:
        # compare timings
        speedup = timing_cvxpy / timing_ac
        print(f"speedup: acados vs cvxpy (gen + layer): {speedup:.3f}")



def create_batched_problem_data(problem: ControlBoundedLqrProblem, n_batch):
    N_horizon = problem.N_horizon

    # cost
    QR_mat = scipy.linalg.block_diag(problem.cost.Q, problem.cost.R)
    c = np.concatenate([problem.cost.q, problem.cost.r])
    H_batch = np.tile(QR_mat, (N_horizon+1, n_batch, 1, 1))
    c_batch = np.tile(c, (N_horizon+1, n_batch, 1))

    # dynamics
    A_batch = np.tile(problem.dynamics.A, (N_horizon+1, n_batch, 1, 1))
    B_batch = np.tile(problem.dynamics.B, (N_horizon+1, n_batch, 1, 1))
    b_batch = np.tile(problem.dynamics.b, (N_horizon+1, n_batch, 1))

    # constraints
    u_lower_batch = np.tile(problem.control_bounds.u_lower, (N_horizon+1, n_batch, 1))
    u_upper_batch = np.tile(problem.control_bounds.u_upper, (N_horizon+1, n_batch, 1))

    return H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch


def get_results_filename(umax, nx, nu, n_batch, N_horizon, tol, solver, sensitivity=False):
    return f"results/{solver}_umax_{umax}_nx{nx}_nu{nu}_nbatch{n_batch}_Nhorizon{N_horizon}_tol{tol}_sensitivity{sensitivity}.npz"

def store_results(x, u, timing, solver, umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False):
    result_filename = get_results_filename(umax, nx, nu, n_batch, N_horizon, tol, solver, sensitivity=sensitivity)
    np.savez(result_filename, x=x, u=u, timing=timing)

def load_results_maybe(solver, umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False):
    result_filename = get_results_filename(umax, nx, nu, n_batch, N_horizon, tol, solver, sensitivity=sensitivity)
    if not os.path.exists(result_filename):
        print(f"File {result_filename} does not exist.")
        return None
    results = np.load(result_filename)
    return results

def get_num_threads_from_multiprocessing():
    import multiprocessing
    num_threads = multiprocessing.cpu_count()
    return num_threads

def control_bounded_lqr_solve_timings(problem: ControlBoundedLqrProblem, x0, with_mpc_pytorch = True, with_mpc_pytorch_cuda = True, with_cvxpy=True, with_cvxpy_cuda=True, with_acados = True, num_threads=None, codegen_suff="", n_exec=1):
    if with_mpc_pytorch_cuda:
        assert with_mpc_pytorch

    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    # get dimensions
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon
    umax = problem.control_bounds.u_upper[0]
    tol = problem.tol

    # dimensions
    n_batch = x0.shape[0]

    # solve with acados
    if with_acados:
        x_ac, u_ac, timing_ac, _ = solve_using_acados(problem, x0, batched=True, num_threads=num_threads, n_exec=n_exec)
        store_results(x_ac, u_ac, timing_ac, "acados", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)

    if with_mpc_pytorch:
        # create batched problem data
        H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

        # solve using MPC class of mpc.pytorch
        x_mpytorch, u_mpytorch, timing_mpytorch, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, tol, n_exec=n_exec)
        store_results(x_mpytorch, u_mpytorch, timing_mpytorch, "mpc_pytorch", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)

        # print(f"{u_mpytorch=}")

        if with_mpc_pytorch_cuda:
            # move to GPU
            device = "cuda"
            x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, tol, device=device, n_exec=n_exec)
            store_results(x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, "mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)

    if with_cvxpy:
        # create batched problem data
        H_batch, _,  A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = (
            create_batched_problem_data(problem, n_batch)
        )

        # solve using cvxpygen + cvxpylayer
        x_cvxpy, u_cvxpy, timing_cvxpy, _ = solve_using_cvxpy(
            H_batch,
            A_batch,
            B_batch,
            b_batch,
            x0,
            u_lower_batch,
            u_upper_batch,
            N_horizon,
            tol,
            codegen_suff=codegen_suff,
            n_exec=n_exec
        )
        store_results(
            x_cvxpy,
            u_cvxpy,
            timing_cvxpy,
            "cvxpy",
            umax,
            nx,
            nu,
            n_batch,
            N_horizon, tol,
            sensitivity=False,
        )

        if with_cvxpy_cuda:
            # move to GPU
            device = "cuda"
            x_cvxpy_cuda, u_cvxpy_cuda, timing_cvxpy_cuda, _ = solve_using_cvxpy(
                H_batch,
                A_batch,
                B_batch,
                b_batch,
                x0,
                u_lower_batch,
                u_upper_batch,
                N_horizon,
                tol,
                device=device,
                codegen_suff=codegen_suff,
                n_exec=n_exec
            )
            store_results(
                x_cvxpy_cuda,
                u_cvxpy_cuda,
                timing_cvxpy_cuda,
                "cvxpy_cuda",
                umax,
                nx,
                nu,
                n_batch,
                N_horizon, tol,
                sensitivity=False,
            )

    print("Timings:")
    if with_acados:
        print(f"acados: {timing_ac=}")
    if with_mpc_pytorch:
        print(f"mpc-pytorch: {timing_mpytorch=}")
    if with_cvxpy:
        print(f"cvxpy (gen + layer): {timing_cvxpy=}")


N_BATCH_EXPERIMENT = 2**7

PROBLEM_CONFIGS = [
    (1e4, 8, 4, 20, 1e-6, "default"),
    (1e0, 8, 4, 20, 1e-6, "umax"),
    (1e4, 8, 4, 100, 1e-6, "N_horizon_big"),
    # (1e4, 8, 4, 20, 1e-4, "tolerance_big"),
    # (1e4, 8, 4, 20, 1e-8, "tolerance_small"),
    (1e4, 16, 8, 20, 1e-6, "high_dim"),
    # (1e4, 32, 16, 20, 1e-6, "high_dim_very"),
]

RUN_CONFIGS = PROBLEM_CONFIGS

def main_experiment(with_acados=True, with_mpc_pytorch=True, with_mpc_pytorch_cuda=True, with_cvxpy=True, with_cvxpy_cuda=True, num_threads=None, n_exec=1):
    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    for umax, nx, nu, N_horizon, tol, codegen_suff in RUN_CONFIGS:
        print(f"running with {codegen_suff}")
        x0 = 0.5 * npr.randn(N_BATCH_EXPERIMENT, nx)
        # load x0 from file to get timings consistent with initial submission.
        # NOTE: for new sampled x0, mpc.pytorch is roughly 2x slower.
        # x0 = np.load("x0_initial_submission.npy")
        problem = define_bounded_lqr_test_problem(umax=umax, nx=nx, nu=nu, N=N_horizon, tol=tol)
        control_bounded_lqr_solve_timings(problem, x0=x0, with_acados=with_acados, with_mpc_pytorch=with_mpc_pytorch, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, with_cvxpy=with_cvxpy, with_cvxpy_cuda=with_cvxpy_cuda, num_threads=num_threads, codegen_suff=codegen_suff, n_exec=n_exec)
        control_bounded_lqr_solve_and_adj_sens_timings(problem, x0=x0, with_acados=with_acados, with_mpc_pytorch=with_mpc_pytorch, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, with_cvxpy=with_cvxpy, with_cvxpy_cuda=with_cvxpy_cuda, num_threads=num_threads, codegen_suff=codegen_suff, n_exec=n_exec)

def speedup_formatter(timing, baseline):
    string = f"{timing/baseline:.2g}"
    if "e+0" in string:
        string = f"{int(float(string))}"
    string = f"$\\times{string}$"
    return string


def evaluate_experiment_latex(cuda: bool = False):
    n_batch = N_BATCH_EXPERIMENT
    configs = RUN_CONFIGS

    # --- Default config (first entry) for computing differences ---
    default_config = configs[0]
    default_umax, default_nx, default_nu, default_N, default_tol, _ = default_config

    table_string = r"\begin{table*}"
    table_string += "\n\centering\n"
    table_string += "\caption{Timings in [ms] for solving $n_{\mathrm{batch}} \!=\! " + f"{n_batch}$ bounded LQR problems with varying parameters."
    table_string += r" In the config column, the difference to the default configuration is shown."
    table_string += r" In parentheses are multiples of the \acados{} runtime."
    table_string += r"\label{tab:mcp_pytorch_big}" + "\n"
    table_string += "}\n"
    table_string += "\\vspace{-1.5mm}\n"
    table_string += "\\footnotesize\n"
    table_string += r"\begin{tabular}{ccccccc}" + "\n"
    table_string += r"\toprule" + "\n"
    table_string += r"& \multicolumn{3}{c}{\textbf{Nominal solution}} & \multicolumn{3}{c}{\textbf{Solution + adjoint sens.}}\\" + "\n"
    table_string += r"config & \acados & \texttt{mpc.pytorch} & \texttt{cvxpygen}"
    table_string += r"& \acados & \texttt{mpc.pytorch} & \texttt{cvxpygen}"
    table_string += "\\\\\n"
    table_string += r"\midrule" + "\n"

    for i, (umax, nx, nu, N_horizon, tol, _) in enumerate(configs):
        # --- Config column (difference from default) ---
        if i == 0:
            config_str = "default"
        else:
            diff_parts = []
            if umax != default_umax:
                umax_log = math.log10(umax)
                if umax_log == int(umax_log):
                    umax_exp = int(umax_log)
                    if umax_exp == 0:
                        diff_parts.append(r"$u_{\max} = 1$")
                    else:
                        diff_parts.append(f"$u_{{\\max}} = 10^{{{umax_exp}}}$")
                else:
                    diff_parts.append(f"$u_{{\\max}} = {umax}$")
            if nx != default_nx:
                diff_parts.append(f"$n_x = {nx}$")
            if nu != default_nu:
                diff_parts.append(f"$n_u = {nu}$")
            if N_horizon != default_N:
                diff_parts.append(f"$N = {N_horizon}$")
            if tol != default_tol:
                tol_exp = int(math.log10(tol))
                diff_parts.append(f"$\\mathrm{{tol}} = 10^{{{tol_exp}}}$")
            config_str = ", ".join(diff_parts)

        # timings solve
        results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
        if not cuda:
            results_mpytorch = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
            results_cpgen = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
        else:
            results_mpytorch = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
            results_cpgen = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
        timing_ac = results_acados['timing']
        timing_mpytorch = results_mpytorch['timing']
        timing_cpgen = results_cpgen['timing']

        table_string += f"{config_str}"
        table_string += f"& {timing_ac*1e3:.1f} & ${int(timing_mpytorch*1e3)} \\;\\,$ ({speedup_formatter(timing_mpytorch, timing_ac)}) & ${int(timing_cpgen*1e3)} \\;\\,$ ({speedup_formatter(timing_cpgen, timing_ac)})"

        # timings sensitivity
        results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
        timing_ac = results_acados['timing']
        if not cuda:
            results_mpytorch = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
            results_cpgen = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
        else:
            results_mpytorch = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
            results_cpgen = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)

        timing_cpgen = results_cpgen['timing']
        timing_mpytorch = results_mpytorch['timing']
        table_string += f"& {timing_ac*1e3:.1f} & ${int(timing_mpytorch*1e3)} \\;\\,$ ({speedup_formatter(timing_mpytorch, timing_ac)}) & ${int(timing_cpgen*1e3)} \\;\\,$ ({speedup_formatter(timing_cpgen, timing_ac)})"

        table_string += "\\\\\n"

    table_string += r"\bottomrule" + "\n"
    table_string += r"\end{tabular}" + "\n"
    table_string += r"\vspace{-2mm}" + "\n"
    table_string += r"\end{table*}" + "\n"

    print(table_string)
    name = "results_table.tex" if not cuda else "results_table_cuda.tex"
    with open(name, "w") as f:
        f.write(table_string)
        print(f"Written to {name}")


def evaluate_experiment_markdown(cuda: bool = False):
    """
    Evaluates the experiment with varying N_horizon, tolerance, and dimensions
    and generates a results table in Markdown format.
    """
    n_batch = N_BATCH_EXPERIMENT
    configs = RUN_CONFIGS

    table_string = ""

    # --- Generate caption ---
    caption_base = f"Timings for solving $n_{{\\mathrm{{batch}}}} = {n_batch}$ bounded LQR problems with varying parameters and defaults $N=20$, $n_x=8$, $n_u=$ and $n_\\theta = n_x^2+n_xn_u + n_x + (n_x+n_u)^2$. In the config column, the difference to the default configuration is shown. In parentheses are multiples of the `acados` runtime."
    table_string += f"**Table:** {caption_base}\n\n"

    # --- Generate Table Header ---
    table_string += "| config | `acados` (Nominal) | `mpc.pytorch` (Nominal) | `cvxpygen` (Nominal) | `acados` (Nominal + Adjoint Sens.) | `mpc.pytorch` (Nominal + Adjoint Sens.) | `cvxpygen` (Nominal + Adjoint Sens.) |\n"
    table_string += "|-----------------|-------------------|------------------------|---------------------|-------------------------------|----------------------------------|-----------------------------------|\n"

    # --- Default config (first entry) for computing differences ---
    default_config = configs[0]
    default_umax, default_nx, default_nu, default_N, default_tol, _ = default_config

    # --- Generate Table Rows ---
    for i, (umax, nx, nu, N_horizon, tol, _) in enumerate(configs):
        row_parts = []

        # --- Config Column (difference from default) ---
        if i == 0:
            config_str = "default"
        else:
            diff_parts = []
            if umax != default_umax:
                umax_log = math.log10(umax)
                if umax_log == int(umax_log):
                    umax_exp = int(umax_log)
                    if umax_exp == 0:
                        diff_parts.append(r"$u_{\max} = 1$")
                    else:
                        diff_parts.append(f"$u_{{\\max}} = 10^{{{umax_exp}}}$")
                else:
                    diff_parts.append(f"$u_{{\\max}} = {umax}$")
            if nx != default_nx:
                diff_parts.append(f"$n_x = {nx}$")
            if nu != default_nu:
                diff_parts.append(f"$n_u = {nu}$")
            if N_horizon != default_N:
                diff_parts.append(f"$N = {N_horizon}$")
            if tol != default_tol:
                tol_exp = int(math.log10(tol))
                diff_parts.append(f"$\\mathrm{{tol}} = 10^{{{tol_exp}}}$")
            config_str = ", ".join(diff_parts)
        row_parts.append(config_str)

        # --- Timings: nominal solution ---
        results_acados_nom = load_results_maybe("acados", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
        if not cuda:
            results_mpytorch_nom = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
            results_cpgen_nom = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
        else:
            results_mpytorch_nom = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
            results_cpgen_nom = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)

        timing_ac_nom = results_acados_nom['timing'] if results_acados_nom else float('nan')
        timing_mpytorch_nom = results_mpytorch_nom['timing'] if results_mpytorch_nom else float('nan')
        timing_cpgen_nom = results_cpgen_nom['timing'] if results_cpgen_nom else float('nan')

        row_parts.extend([f"{timing_ac_nom*1e3:.1f}" if not math.isnan(timing_ac_nom) else "-",
                          f"{int(timing_mpytorch_nom*1e3)} ({speedup_formatter(timing_mpytorch_nom, timing_ac_nom)})" if not math.isnan(timing_mpytorch_nom) else "-",
                          f"{int(timing_cpgen_nom*1e3)} ({speedup_formatter(timing_cpgen_nom, timing_ac_nom)})" if not math.isnan(timing_cpgen_nom) else "-"])

        # --- Timings: solution + adjoint sensitivity ---
        results_acados_sens = load_results_maybe("acados", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
        if not cuda:
            results_mpytorch_sens = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
            results_cpgen_sens = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
        else:
            results_mpytorch_sens = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
            results_cpgen_sens = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=True)
        timing_ac_sens = results_acados_sens['timing'] if results_acados_sens else float('nan')
        timing_mpytorch_sens = results_mpytorch_sens['timing'] if results_mpytorch_sens else float('nan')
        timing_cpgen_sens = results_cpgen_sens['timing'] if results_cpgen_sens else float('nan')

        print(f"\nTimings for nx={nx}, nu={nu}, N={N_horizon}, tol={tol}")
        print(f"Nominal: timings in ms: acados: {timing_ac_nom*1e3:.2f}, mpc.pytorch: {timing_mpytorch_nom*1e3:.2f}, cvxpygen: {timing_cpgen_nom*1e3:.2f}")
        print(f"Sensitivity: timings in ms: acados: {timing_ac_sens*1e3:.2f}, mpc.pytorch: {timing_mpytorch_sens*1e3:.2f}, cvxpygen: {timing_cpgen_sens*1e3:.2f}")

        row_parts.extend([f"{timing_ac_sens*1e3:.1f}" if not math.isnan(timing_ac_sens) else "-",
                          f"{int(timing_mpytorch_sens*1e3)} ({speedup_formatter(timing_mpytorch_sens, timing_ac_sens)})" if not math.isnan(timing_mpytorch_sens) else "-",
                          f"{int(timing_cpgen_sens*1e3)} ({speedup_formatter(timing_cpgen_sens, timing_ac_sens)})" if not math.isnan(timing_cpgen_sens) else "-"])

        # --- Append Row to Table String ---
        table_string += "| " + " | ".join(row_parts) + " |\n"

    # --- Write to File ---
    name = "results_table.md" if not cuda else "results_table_cuda.md"
    with open(name, "w") as f:
        f.write(table_string)
        print(f"Written to {name}")


def analyze_constraint_activeness_in_results(config):
    umax, nx, nu, N_horizon, tol, _ = config
    n_batch = N_BATCH_EXPERIMENT
    results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
    u_sol = results_acados['u']
    u_max_abs = np.max(np.abs(u_sol))
    print(f"u_max_abs: {u_max_abs}, umax: {umax}")
    if u_max_abs < 0.99 * umax:
        print("control bounds are always INACTIVE")
    else:
        # count number of active bounds
        u_bu_active = (u_sol >= 0.99*umax).sum()
        u_lb_active = (u_sol <= -0.99*umax).sum()
        print(f"u_lb_active: {u_lb_active}, u_bu_active: {u_bu_active}")
        print(f"total active bounds: {u_lb_active + u_bu_active}")
        print(f"ratio of active constraints {100 * (u_lb_active + u_bu_active) / u_sol.size :.3f} %")


def compare_results_xu(config):
    umax, nx, nu, N_horizon, tol, _ = config
    n_batch = N_BATCH_EXPERIMENT
    solver_names = ["acados", "mpc_pytorch", "cvxpy"]
    u_sol_list = []
    x_sol_list = []
    for solver in solver_names:
        results = load_results_maybe(solver, umax, nx, nu, n_batch, N_horizon, tol, sensitivity=False)
        if results is None:
            raise Exception(f"Results for {solver} not found.")
        u_sol_list.append(results['u'])
        x_sol_list.append(results['x'])

    u_sol_ref = u_sol_list[0]
    x_sol_ref = x_sol_list[0]

    for i in range(1, len(u_sol_list)):
        diff_u = np.abs(u_sol_list[i] - u_sol_ref)
        diff_x = np.abs(x_sol_list[i] - x_sol_ref)
        max_diff_u = np.max(diff_u)
        mean_diff_u = np.mean(diff_u)
        max_diff_x = np.max(diff_x)
        mean_diff_x = np.mean(diff_x)
        print(f"\nComparing {solver_names[0]} with {solver_names[i]}:")
        print(f"Config {umax}, {nx}, {nu}, {N_horizon}, {tol}")
        print(f"Diff x: max {max_diff_x:.4f}, mean: {mean_diff_x:.4f}, diff u: max {max_diff_u:.4f}, mean: {mean_diff_u:.4f}")
        if max_diff_x > 1e-4 or max_diff_u > 1e-4:
            print("Results are NOT consistent.")
        else:
            print("Results are consistent within numerical precision.")


if __name__ == "__main__":
    # ensure results directory exists
    os.makedirs("results", exist_ok=True)

    num_threads = 4  # experiments in paper were run with this for the i7-8665U CPU
    # num_threads = 16  # experiments in paper were run with this for the AMD Ryzen 9 5950X 16-Core Processor
    # num_threads = 8  # -> detect number of threads automatically.

    # if cuda is available, use it (or set this manually to True/False)
    cuda = False

    # experiments
    # main_experiment(with_acados=True, with_mpc_pytorch=True, with_mpc_pytorch_cuda=cuda, with_cvxpy=True, with_cvxpy_cuda=cuda, num_threads=num_threads, n_exec=5)

    # Use this to generate the desired LaTeX tables out of the experiment data
    evaluate_experiment_latex(cuda=cuda)
    evaluate_experiment_markdown(cuda=cuda)

    analyze_constraint_activeness_in_results(RUN_CONFIGS[0])
    analyze_constraint_activeness_in_results(RUN_CONFIGS[1])
    for i in range(len(RUN_CONFIGS)):
        compare_results_xu(RUN_CONFIGS[i])
