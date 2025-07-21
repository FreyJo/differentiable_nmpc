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

TOL = 1e-6
N_HORIZON = 20


def define_bounded_lqr_test_problem(nx=8, nu=4, umax=1.0):

    Q_mat = np.eye(nx)
    R_mat = np.eye(nu)
    quadratic_cost = QuadraticCost(Q_mat, R_mat, np.zeros(nx), np.zeros(nu))

    alpha = 0.2
    A_mat = np.eye(nx)+alpha*npr.randn(nx, nx)
    B_mat = npr.randn(nx, nu)
    b = npr.randn(nx)

    dynamics = LinearDiscreteDynamics(A_mat, B_mat, b)
    control_bounds = ControlBounds(-umax*np.ones((nu,)), umax*np.ones((nu,)))

    ocp = ControlBoundedLqrProblem(dynamics, quadratic_cost, control_bounds, N_horizon=N_HORIZON)
    return ocp

def solve_using_cvxpy(
    H_batch,
    A_batch,
    B_batch,
    b_batch,
    x0,
    u_lower_batch,
    u_upper_batch,
    seed=None,
    device="cpu",
    codegen_suff=""
):
    """Solve min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t
    s.t. x_{t+1} = A_t x_t + B_t u_t + b_t
         x_0 = x0
         u_lower <= u <= u_upper
    """

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
        N_horizon = N_HORIZON + 1
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
        "eps_abs": TOL,
        "eps_rel": TOL,
        "max_iter": 100000,
    }

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

    return (
        x_cp_tch.cpu().detach().numpy(),
        u_cp_tch.cpu().detach().numpy(),
        timing,
        du_dp,
    )

def solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, lbu, ubu,
                            seed=None, device="cpu"):
    
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

    dynamics = AffineDynamics(A_batch, B_batch, b_batch)
    quad_cost = QuadCost(H_batch, c_batch)
    mpc_pytorch_solver = mpc.MPC(
        nx, nu, N_horizon, lbu, ubu, None,
        lqr_iter=100,  # NOTE: defines max iterations for LQR solver, with the default of 10, it fails on many more problems.
        backprop=(seed is not None),
        exit_unconverged=False,
        eps=TOL,
        n_batch=n_batch,
    )

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

    x_mpytorch = x_mpytorch.cpu().detach().numpy()
    u_mpytorch = u_mpytorch.cpu().detach().numpy()
    # if du_dp_adj is not None:
    #     du_dp_adj = du_dp_adj.cpu()

    return x_mpytorch, u_mpytorch, timing, du_dp_adj


def control_bounded_lqr_solve_and_adj_sens_timings(problem: ControlBoundedLqrProblem, x0, with_mpc_pytorch = False, with_mpc_pytorch_cuda = False, with_cvxpy = False, with_cvxpy_cuda = False, num_threads=None, codegen_suff=""):
    if with_mpc_pytorch_cuda:
        assert with_mpc_pytorch
    if with_cvxpy_cuda:
        assert with_cvxpy_cuda

    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()
    with_acados = True

    # get dimensions
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon

    # define seed
    seed = np.ones((nu,))

    # dimensions
    n_batch = x0.shape[0]

    # solve with acados
    if with_acados:
        x_ac, u_ac, timing_ac, du_dp_adj_batch = solve_using_acados(problem, x0, seed=seed, batched=True, num_threads=num_threads)
        store_results(x_ac, u_ac, timing_ac, "acados", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, sensitivity=True)

    if with_mpc_pytorch:
        # create batched problem data
        H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

        # solve using MPC class of mpc.pytorch
        x_mpytorch, u_mpytorch, timing_mpytorch, du_dp_adj_mpy = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, seed=seed)
        store_results(x_mpytorch, u_mpytorch, timing_mpytorch, "mpc_pytorch", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, sensitivity=True)
        # print(f"{u_mpytorch=}")

        if with_mpc_pytorch_cuda:
            # move to GPU
            device = "cuda"
            x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, du_dp_adj_mpy_cuda = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, seed=seed, device=device)
            store_results(x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, "mpc_pytorch_cuda", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, sensitivity=True)

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
            seed=seed,
            codegen_suff=codegen_suff
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
                seed=seed,
                device=device,
                codegen_suff=codegen_suff
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


def get_results_filename(umax, nx, nu, n_batch, N_horizon, solver, sensitivity=False):
    return f"results/{solver}_umax_{umax}_nx{nx}_nu{nu}_nbatch{n_batch}_Nhorizon{N_horizon}_sensitivity{sensitivity}.npz"

def store_results(x, u, timing, solver, umax, nx, nu, n_batch, N_horizon, sensitivity=False):
    result_filename = get_results_filename(umax, nx, nu, n_batch, N_horizon, solver, sensitivity=sensitivity)
    np.savez(result_filename, x=x, u=u, timing=timing)

def load_results_maybe(solver, umax, nx, nu, n_batch, N_horizon, sensitivity=False):
    result_filename = get_results_filename(umax, nx, nu, n_batch, N_horizon, solver, sensitivity=sensitivity)
    if not os.path.exists(result_filename):
        print(f"File {result_filename} does not exist.")
        return None
    results = np.load(result_filename)
    return results

def get_num_threads_from_multiprocessing():
    import multiprocessing
    num_threads = multiprocessing.cpu_count()
    return num_threads

def control_bounded_lqr_solve_timings(problem: ControlBoundedLqrProblem, x0, with_mpc_pytorch = True, with_mpc_pytorch_cuda = True, with_cvxpy=True, with_cvxpy_cuda=True, with_acados = True, num_threads=None, codegen_suff=""):
    if with_mpc_pytorch_cuda:
        assert with_mpc_pytorch

    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    # get dimensions
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon
    umax = problem.control_bounds.u_upper[0]

    # dimensions
    n_batch = x0.shape[0]

    # solve with acados
    if with_acados:
        x_ac, u_ac, timing_ac, _ = solve_using_acados(problem, x0, batched=True, num_threads=num_threads)
        store_results(x_ac, u_ac, timing_ac, "acados", umax, nx, nu, n_batch, N_horizon, sensitivity=False)

    if with_mpc_pytorch:
        # create batched problem data
        H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

        # solve using MPC class of mpc.pytorch
        x_mpytorch, u_mpytorch, timing_mpytorch, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch)
        store_results(x_mpytorch, u_mpytorch, timing_mpytorch, "mpc_pytorch", umax, nx, nu, n_batch, N_horizon, sensitivity=False)

        # print(f"{u_mpytorch=}")

        if with_mpc_pytorch_cuda:
            # move to GPU
            device = "cuda"
            x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, device=device)
            store_results(x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, "mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, sensitivity=False)

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
            codegen_suff=codegen_suff,
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
            N_horizon,
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
                device=device,
                codegen_suff=codegen_suff,
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
                N_horizon,
                sensitivity=False,
            )

    print("Timings:")
    if with_acados:
        print(f"acados: {timing_ac=}")
    if with_mpc_pytorch:
        print(f"mpc-pytorch: {timing_mpytorch=}")
    if with_cvxpy:
        print(f"cvxpy (gen + layer): {timing_cvxpy=}")



PROBLEM_CONFIGS = [(1e4, 8, 4, "one"),
                   (1e0, 8, 4, "two")]
N_BATCH_EXPERIMENT = 2**7
def main_experiment(with_mpc_pytorch=True, with_mpc_pytorch_cuda=True, with_cvxpy=True, with_cvxpy_cuda=True, num_threads=None):
    n_batch = N_BATCH_EXPERIMENT
    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    for umax, nx, nu, codegen_suff in PROBLEM_CONFIGS:
        # x0 = npr.randn(n_batch, nx)
        # load x0 from file to get timings consistent with initial submission.
        # NOTE: for new sampled x0, mpc.pytorch is roughly 2x slower.
        x0 = np.load("x0_initial_submission.npy")
        problem = define_bounded_lqr_test_problem(umax=umax, nx=nx, nu=nu)
        control_bounded_lqr_solve_timings(problem, x0=x0, with_mpc_pytorch=with_mpc_pytorch, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, with_cvxpy=with_cvxpy, with_cvxpy_cuda=with_cvxpy_cuda, num_threads=num_threads, codegen_suff=codegen_suff)
        control_bounded_lqr_solve_and_adj_sens_timings(problem, x0=x0, with_mpc_pytorch=with_mpc_pytorch, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, with_cvxpy=with_cvxpy, with_cvxpy_cuda=with_cvxpy_cuda, num_threads=num_threads, codegen_suff=codegen_suff)

def speedup_formatter(timing, baseline):
    string = f"{timing/baseline:.2g}"
    if "e+0" in string:
        string = f"{int(float(string))}"
    string = f"$\\times{string}$"
    return string

def evaluate_experiment_latex(cuda: bool = False):
    n_batch = N_BATCH_EXPERIMENT
    configs = PROBLEM_CONFIGS

    nx_vals = [c[1] for c in configs]
    nu_vals = [c[2] for c in configs]
    nxu_varies = len(set(nx_vals)) != 1 or len(set(nu_vals)) != 1

    table_string = r"\begin{table*}"
    table_string += "\n\centering\n"
    table_string += "\caption{Timings in [ms] for solving $n_{\mathrm{batch}} \!=\! " + f"{n_batch}$ bounded LQR problems with $N \!=\! {N_HORIZON}$"
    if not nxu_varies:
        nx = configs[0][1]
        nu = configs[0][2]
        table_string += f", $n_x \!=\! {nx}$, $n_u \!=\! {nu}$, $n_\\theta" + f" \!= \!{nx*(nx+nu+1)+(nx+nu)**2}$"
    table_string += r". In parenthesis in multiples the \acados{} runtime for the others."

    table_string += r"\label{tab:mcp_pytorch}" + "\n"
    table_string += "}\n"
    table_string += "\\vspace{-1.5mm}\n"
    table_string += "\\footnotesize\n"
    table_string += r"\begin{tabular}{ccccccc}" + "\n"
    table_string += r"\toprule" + "\n"
    table_string += r"& \multicolumn{3}{c}{\textbf{Nominal solution}} & \multicolumn{3}{c}{\textbf{Solution + adjoint sens.}}\\" + "\n"
    table_string += r"$u_{\mathrm{max}}$ & \acados & \texttt{mpc.pytorch} & \texttt{cvxpygen}"
    table_string += r"& \acados & \texttt{mpc.pytorch} & \texttt{cvxpygen}"
    table_string += "\\\\\n"
    table_string += r"\midrule" + "\n"

    for umax, nx, nu, _ in configs:
        # timings solve
        results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        if not cuda:
            results_mpytorch = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
            results_cpgen = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        else:
            results_mpytorch = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
            results_cpgen = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        timing_ac = results_acados['timing']
        timing_mpytorch = results_mpytorch['timing']
        timing_cpgen = results_cpgen['timing']
        table_string += "$"
        if umax == 1e4:
            table_string += r"10^4"
        else:
            table_string += f"{umax}"
        table_string += "$"
        if nxu_varies:
            table_string += f", $n_x = {nx}$, $n_u= {nu}$"
        table_string += f"& {timing_ac*1e3:.1f} & ${int(timing_mpytorch*1e3)} \;\,$ ({speedup_formatter(timing_mpytorch, timing_ac)}) & ${int(timing_cpgen*1e3)} \;\,$ ({speedup_formatter(timing_cpgen, timing_ac)})"

        # timings sensitivity
        results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        timing_ac = results_acados['timing']
        if not cuda:
            results_mpytorch = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
            results_cpgen = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        else:
            results_mpytorch = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
            results_cpgen = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)

        timing_cpgen = results_cpgen['timing']
        timing_mpytorch = results_mpytorch['timing']
        # table_string += f"& {timing_ac*1e3:.1f} & {speedup_formatter(timing_mpytorch, timing_ac)} & {speedup_formatter(timing_cpgen, timing_ac)}"
        table_string += f"& {timing_ac*1e3:.1f} & ${int(timing_mpytorch*1e3)} \;\,$ ({speedup_formatter(timing_mpytorch, timing_ac)}) & ${int(timing_cpgen*1e3)} \;\,$ ({speedup_formatter(timing_cpgen, timing_ac)})"


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
    Evaluates the experiment and generates a results table in Markdown format.
    """
    n_batch = N_BATCH_EXPERIMENT
    configs = PROBLEM_CONFIGS

    nx_vals = [c[1] for c in configs]
    nu_vals = [c[2] for c in configs]
    nxu_varies = len(set(nx_vals)) != 1 or len(set(nu_vals)) != 1

    table_string = ""

    # --- Generate caption ---
    caption_base = f"Timings for solving $n_{{\mathrm{{batch}}}} = {n_batch}$ bounded LQR problems with $N = {N_HORIZON}$"
    if not nxu_varies:
        nx = configs[0][1]
        nu = configs[0][2]
        n_theta = nx * (nx + nu + 1) + (nx + nu)**2
        caption_base += f", $n_x = {nx}$, $n_u = {nu}$, $n_\\theta = {n_theta}$"
    caption_base += ". Given in [ms] for `acados` and in multiples of the `acados` runtime for the others."
    table_string += f"**Table:** {caption_base}\n\n" # Add caption as bold text

    # --- Generate Table Header ---
    table_string += "| $u_{\mathrm{max}}$ | `acados` (Nominal) | `mpc.pytorch` (Nominal) | `cvxpygen` (Nominal) | `acados` (Adjoint Sens.) | `mpc.pytorch` (Adjoint Sens.) | `cvxpygen` (Adjoint Sens.) |\n"
    table_string += "|---------------|------------------|-----------------------|-------------------|-------------------------|--------------------------|-------------------------|\n"

    # --- Generate Table Rows ---
    for umax, nx, nu, _ in configs:
        row_parts = []

        # --- Config Column ---
        config_str = "$"
        if umax == 1e4:
            config_str += r"10^4"
        else:
            config_str += f"{umax:.1f}" if isinstance(umax, float) else f"{umax}" # Format float umax
        config_str += "$"
        if nxu_varies:
            config_str += f", $n_x = {nx}$, $n_u = {nu}$"

        row_parts.append(config_str)

        # --- Timings: nominal solution ---
        results_acados_nom = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        if not cuda:
            results_mpytorch_nom = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
            results_cpgen_nom = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        else:
            results_mpytorch_nom = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
            results_cpgen_nom = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)

        timing_ac_nom = results_acados_nom['timing'] if results_acados_nom else float('nan')
        timing_mpytorch_nom = results_mpytorch_nom['timing'] if results_mpytorch_nom else float('nan')
        timing_cpgen_nom = results_cpgen_nom['timing'] if results_cpgen_nom else float('nan')

        speedup_mpytorch_nom = timing_mpytorch_nom / timing_ac_nom
        speedup_cpgen_nom = timing_cpgen_nom / timing_ac_nom

        row_parts.extend([f"{timing_ac_nom*1e3:.1f}" if not math.isnan(timing_ac_nom) else "-",
                          f"{speedup_mpytorch_nom:.2g}" if not math.isnan(speedup_mpytorch_nom) else "-",
                          f"{speedup_cpgen_nom:.2g}" if not math.isnan(speedup_cpgen_nom) else "-"])


        # --- Timings: solution + adjoint sensitivity ---
        results_acados_sens = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        if not cuda:
            results_mpytorch_sens = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
            results_cpgen_sens = load_results_maybe("cvxpy", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        else:
            results_mpytorch_sens = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
            results_cpgen_sens = load_results_maybe("cvxpy_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        timing_ac_sens = results_acados_sens['timing'] if results_acados_sens else float('nan')
        timing_mpytorch_sens = results_mpytorch_sens['timing'] if results_mpytorch_sens else float('nan')
        timing_cpgen_sens = results_cpgen_sens['timing'] if results_cpgen_sens else float('nan')

        print("\nTimings for umax =", umax)
        print(f"Nominal: timings in ms: acados: {timing_ac_nom*1e3:.2f}, mpc.pytorch: {timing_mpytorch_nom*1e3:.2f}, cvxpygen: {timing_cpgen_nom*1e3:.2f}")
        print(f"Sensitivity: timings in ms: acados: {timing_ac_sens*1e3:.2f}, mpc.pytorch: {timing_mpytorch_sens*1e3:.2f}, cvxpygen: {timing_cpgen_sens*1e3:.2f}")

        # Speedup calculation: mpc.pytorch / acados
        speedup_mpytorch_sens = timing_mpytorch_sens / timing_ac_sens
        speedup_cpgen_sens = timing_cpgen_sens / timing_ac_sens

        row_parts.extend([f"{timing_ac_sens*1e3:.1f}" if not math.isnan(timing_ac_sens) else "-",
                          f"{speedup_mpytorch_sens:.2g}" if not math.isnan(speedup_mpytorch_sens) else "-",
                          f"{speedup_cpgen_sens:.2g}" if not math.isnan(speedup_cpgen_sens) else "-"])


        # --- Append Row to Table String ---
        table_string += "| " + " | ".join(row_parts) + " |\n"

    # --- No Footer needed for Markdown ---

    # --- Write to File ---
    name = "results_table.md" if not cuda else "results_table_cuda.md"
    with open(name, "w") as f:
        f.write(table_string)
        print(f"Written to {name}")


def analyze_constraint_activeness_in_results(config):
    umax, nx, nu, _ = config
    n_batch = N_BATCH_EXPERIMENT
    results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
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
    umax, nx, nu, _ = config
    n_batch = N_BATCH_EXPERIMENT
    solver_names = ["acados", "mpc_pytorch", "cvxpy"]
    u_sol_list = []
    x_sol_list = []
    for solver in solver_names:
        results = load_results_maybe(solver, umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
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
        print(f"Diff x: max {max_diff_x:.4f}, mean: {mean_diff_x:.4f}, diff u: max {max_diff_u:.4f}, mean: {mean_diff_u:.4f}")
        if max_diff_x > TOL*1e2 or max_diff_u > TOL*1e2:
            print("Results are NOT consistent.")
        else:
            print("Results are consistent within numerical precision.")


if __name__ == "__main__":
    # ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # num_threads = 4  # experiments in paper were run with this for the i7-8665U CPU
    # num_threads = 16  # experiments in paper were run with this for the AMD Ryzen 9 5950X 16-Core Processor
    num_threads = None  # -> detect number of threads automatically.

    # if cuda is available, use it (or set this manually to True/False)
    cuda = torch.cuda.is_available()

    # experiments
    main_experiment(with_mpc_pytorch=True, with_mpc_pytorch_cuda=cuda, with_cvxpy=True, with_cvxpy_cuda=cuda, num_threads=num_threads)

    # Use this to generate the desired LaTeX tables out of the experiment data
    evaluate_experiment_latex(cuda=cuda)
    evaluate_experiment_markdown(cuda=cuda)

    analyze_constraint_activeness_in_results(PROBLEM_CONFIGS[0])
    analyze_constraint_activeness_in_results(PROBLEM_CONFIGS[1])
    compare_results_xu(PROBLEM_CONFIGS[0])
    compare_results_xu(PROBLEM_CONFIGS[1])
