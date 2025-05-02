import argparse
import torch
import os
import math
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import cvxpy as cp

import scipy.linalg

from torch.autograd import Variable
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


def solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, lbu, ubu,
                            seed=None, device="cpu"):
    N_horizon += 1

    if len(b_batch.shape) == 3:
        b_batch = b_batch[0, 0]
    elif len(b_batch.shape) in [1, 2]:
        b_batch = b_batch
    if len(A_batch.shape) == 4:
        A_batch = A_batch[0, 0]
    else:
        A_batch = A_batch
    if len(B_batch.shape) == 4:
        B_batch = B_batch[0, 0]
    else:
        B_batch = B_batch

    # move batches to GPU
    def prep(x):
        return x.detach().to(device).requires_grad_()

    A_batch = prep(A_batch)
    B_batch = prep(B_batch)
    b_batch = prep(b_batch)
    H_batch = prep(H_batch)
    c_batch = prep(c_batch)
    x0 = prep(x0)
    lbu = prep(lbu)
    ubu = prep(ubu)

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

    x_mpytorch_, u_mpytorch_, objs_mpytorch = mpc_pytorch_solver(x0, quad_cost, dynamics)

    time_start = timer()
    x_mpytorch_, u_mpytorch_, objs_mpytorch = mpc_pytorch_solver(x0, quad_cost, dynamics)
    x_mpytorch = util.get_data_maybe(x_mpytorch_)
    u_mpytorch = util.get_data_maybe(u_mpytorch_)

    if seed is None:
        du_dp_adj = None
    else:
        loss = torch.sum(u_mpytorch_[0, :, :])
        loss.backward()  # needed when moving to device
        du_dp_adj = np.concatenate((A_batch.grad.cpu().numpy().flatten(order='F'),
                                    B_batch.grad.cpu().numpy().flatten(order='F'),
                                    b_batch.grad.cpu().numpy().flatten(),
                                    H_batch.grad.cpu().numpy().flatten(order='F'),
                                    ))

    timing = timer() - time_start

    # move back to CPU
    x_mpytorch = x_mpytorch.cpu()
    u_mpytorch = u_mpytorch.cpu()
    # if du_dp_adj is not None:
    #     du_dp_adj = du_dp_adj.cpu()

    return x_mpytorch, u_mpytorch, timing, du_dp_adj


def control_bounded_lqr_solve_and_adj_sens_timings(problem: ControlBoundedLqrProblem, n_batch=2**12, with_mpc_pytorch = False, with_mpc_pytorch_cuda = False, num_threads=None):
    if with_mpc_pytorch_cuda:
        assert with_mpc_pytorch

    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()
    with_acados = True

    # get dimensions
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon

    # define seed
    seed = np.ones((nu,))

    # dimensions
    x0 = npr.randn(n_batch, nx)

    # solve with acados
    if with_acados:
        x_ac, u_ac, timing_ac, du_dp_adj_batch = solve_using_acados(problem, x0, seed=seed, batched=True, num_threads=num_threads)
        store_results(x_ac, u_ac, timing_ac, "acados", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, sensitivity=True)

    if with_mpc_pytorch:
        # create batched problem data
        H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

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
        x_mpytorch, u_mpytorch, timing_mpytorch, du_dp_adj_mpy = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, seed=seed)
        store_results(x_mpytorch, u_mpytorch, timing_mpytorch, "mpc_pytorch", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, sensitivity=True)
        # print(f"{u_mpytorch=}")

        if with_mpc_pytorch_cuda:
            # move to GPU
            device = "cuda"
            x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, du_dp_adj_mpy_cuda = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, seed=seed, device=device)
            store_results(x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, "mpc_pytorch_cuda", problem.control_bounds.u_upper[0], nx, nu, n_batch, N_horizon, sensitivity=True)

    print(f"Timings:")
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

def control_bounded_lqr_solve_timings(problem: ControlBoundedLqrProblem, n_batch=2**12, with_mpc_pytorch = True, with_mpc_pytorch_cuda = True, with_acados = True, num_threads=None):
    if with_mpc_pytorch_cuda:
        assert with_mpc_pytorch

    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    # get dimensions
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon
    umax = problem.control_bounds.u_upper[0]

    # dimensions
    x0 = npr.randn(n_batch, nx)

    # solve with acados
    if with_acados:
        x_ac, u_ac, timing_ac, _ = solve_using_acados(problem, x0, batched=True, num_threads=num_threads)
        store_results(x_ac, u_ac, timing_ac, "acados", umax, nx, nu, n_batch, N_horizon, sensitivity=False)

    if with_mpc_pytorch:
        # create batched problem data
        H_batch, c_batch, A_batch, B_batch, b_batch, u_lower_batch, u_upper_batch = create_batched_problem_data(problem, n_batch)

        # convert to torch
        H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch = [
            Variable(torch.Tensor(x).double()) if x is not None else None
            for x in [H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch]
        ]

        # solve using MPC class of mpc.pytorch
        x_mpytorch, u_mpytorch, timing_mpytorch, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch)
        store_results(x_mpytorch, u_mpytorch, timing_mpytorch, "mpc_pytorch", umax, nx, nu, n_batch, N_horizon, sensitivity=False)

        print(f"{u_mpytorch=}")

        if with_mpc_pytorch_cuda:
            # move to GPU
            device = "cuda"
            x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, _ = solve_using_mpc_pytorch(nx, nu, N_horizon, n_batch, H_batch, c_batch, A_batch, B_batch, b_batch, x0, u_lower_batch, u_upper_batch, device=device)
            store_results(x_mpytorch_cuda, u_mpytorch_cuda, timing_mpytorch_cuda, "mpc_pytorch_cuda", umax, nx, nu, n_batch, N_horizon, sensitivity=False)

    print(f"Timings:")
    if with_acados:
        print(f"acados: {timing_ac=}")
    if with_mpc_pytorch:
        print(f"mpc-pytorch: {timing_mpytorch=}")


PROBLEM_CONFIGS = [(1e4, 8, 4), (1e0, 8, 4)]
N_BATCH_EXPERIMENT = 2**7
def main_experiment(with_mpc_pytorch=True, with_mpc_pytorch_cuda=True, num_threads=None):
    n_batch = N_BATCH_EXPERIMENT
    if num_threads is None:
        num_threads = get_num_threads_from_multiprocessing()

    for umax, nx, nu in PROBLEM_CONFIGS:
        problem = define_bounded_lqr_test_problem(umax=umax, nx=nx, nu=nu)
        control_bounded_lqr_solve_timings(problem, n_batch=n_batch, with_mpc_pytorch=with_mpc_pytorch, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, num_threads=num_threads)
        control_bounded_lqr_solve_and_adj_sens_timings(problem, n_batch=n_batch, with_mpc_pytorch=with_mpc_pytorch, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, num_threads=num_threads)

def evaluate_experiment_latex(mpc_pytorch_cuda: bool = False):
    n_batch = N_BATCH_EXPERIMENT
    configs = PROBLEM_CONFIGS

    nx_vals = [c[1] for c in configs]
    nu_vals = [c[2] for c in configs]
    nxu_varies = len(set(nx_vals)) != 1 or len(set(nu_vals)) != 1

    table_string = r"\begin{table*}"
    table_string += "\n\centering\n"
    table_string += "\caption{Timings in [s] for solving $n_{\mathrm{batch}} \!=\! " + f"{n_batch}$ bounded LQR problems with $N \!=\! {N_HORIZON}$"
    if not nxu_varies:
        nx = configs[0][1]
        nu = configs[0][2]
        table_string += f", $n_x \!=\! {nx}$, $n_u \!=\! {nu}$, $n_\\theta" + f" \!= \!{nx*(nx+nu+1)+(nx+nu)**2}$"
    table_string += ".\n"

    table_string += r"\label{tab:mcp_pytorch}" + "\n"
    table_string += "}\n"
    table_string += "\\vspace{-3mm}\n"
    table_string += "\\small\n"
    table_string += r"\begin{tabular}{ccccccc}" + "\n"
    table_string += r"\toprule" + "\n"
    table_string += r"& \multicolumn{3}{c}{\textbf{Nominal solution}} & \multicolumn{3}{c}{\textbf{Solution + adjoint sens.}}\\" + "\n"
    table_string += r"problem config & \acados & \texttt{mpc.pytorch} & speedup"
    table_string += r"& \acados & \texttt{mpc.pytorch} & speedup"
    table_string += "\\\\\n"
    table_string += r"\midrule" + "\n"

    for umax, nx, nu in configs:
        # timings solve
        results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        if not mpc_pytorch_cuda:
            results_mpytorch = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        else:
            results_mpytorch = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        timing_ac = results_acados['timing']
        timing_mpytorch = results_mpytorch['timing']
        table_string += "$u_{\mathrm{max}} = "
        if umax == 1e4:
            table_string += r"10^4"
        else:
            table_string += f"{umax}"
        table_string += "$"
        if nxu_varies:
            table_string += f", $n_x = {nx}$, $n_u= {nu}$"
        table_string += f"& {timing_ac:.3f} & {timing_mpytorch:.3f} & {timing_mpytorch/timing_ac:.2f}"

        # timings sensitivity
        results_acados = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        timing_ac = results_acados['timing']
        if not mpc_pytorch_cuda:
            results_mpytorch = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        else:
            results_mpytorch = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        if results_mpytorch is None:
            table_string += f" & {timing_ac:.2f} & - & - \\\\\n"
        else:
            timing_mpytorch = results_mpytorch['timing']
            table_string += f" & {timing_ac:.3f} & {timing_mpytorch:.3f} & {timing_mpytorch/timing_ac:.2f}"
        table_string += "\\\\\n"

    table_string += r"\bottomrule" + "\n"
    table_string += r"\end{tabular}" + "\n"
    table_string += r"\vspace{-2mm}" + "\n"
    table_string += r"\end{table*}" + "\n"

    print(table_string)
    name = "results_table.tex" if not mpc_pytorch_cuda else "results_table_cuda.tex"
    with open(name, "w") as f:
        f.write(table_string)
        print(f"Written to {name}")


def evaluate_experiment_markdown(mpc_pytorch_cuda: bool = False):
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
    caption_base = f"Timings in [s] for solving $n_{{\mathrm{{batch}}}} = {n_batch}$ bounded LQR problems with $N = {N_HORIZON}$"
    if not nxu_varies:
        nx = configs[0][1]
        nu = configs[0][2]
        n_theta = nx * (nx + nu + 1) + (nx + nu)**2
        caption_base += f", $n_x = {nx}$, $n_u = {nu}$, $n_\\theta = {n_theta}$"
    caption_base += "."
    table_string += f"**Table:** {caption_base}\n\n" # Add caption as bold text

    # --- Generate Table Header ---
    table_string += "| Problem Config | `acados` (Nominal) | `mpc.pytorch` (Nominal) | Speedup (Nominal) | `acados` (Adjoint Sens.) | `mpc.pytorch` (Adjoint Sens.) | Speedup (Adjoint Sens.) |\n"
    table_string += "|---------------|------------------|-----------------------|-------------------|-------------------------|--------------------------|-------------------------|\n"

    # --- Generate Table Rows ---
    for umax, nx, nu in configs:
        row_parts = []

        # --- Config Column ---
        config_str = f"$u_{{\mathrm{{max}}}} = "
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
        if not mpc_pytorch_cuda:
            results_mpytorch_nom = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)
        else:
            results_mpytorch_nom = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=False)

        timing_ac_nom = results_acados_nom['timing'] if results_acados_nom else float('nan')
        timing_mpytorch_nom = results_mpytorch_nom['timing'] if results_mpytorch_nom else float('nan')

        # Speedup calculation: mpc.pytorch / acados
        speedup_nom = timing_mpytorch_nom / timing_ac_nom

        row_parts.extend([f"{timing_ac_nom:.3f}" if not math.isnan(timing_ac_nom) else "-",
                          f"{timing_mpytorch_nom:.2f}" if not math.isnan(timing_mpytorch_nom) else "-",
                          f"{speedup_nom:.2f}" if not math.isnan(speedup_nom) else "-"])


        # --- Timings: solution + adjoint sensitivity ---
        results_acados_sens = load_results_maybe("acados", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        if not mpc_pytorch_cuda:
            results_mpytorch_sens = load_results_maybe("mpc_pytorch", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        else:
            results_mpytorch_sens = load_results_maybe("mpc_pytorch_cuda", umax, nx, nu, n_batch, N_HORIZON, sensitivity=True)
        timing_ac_sens = results_acados_sens['timing'] if results_acados_sens else float('nan')
        timing_mpytorch_sens = results_mpytorch_sens['timing'] if results_mpytorch_sens else float('nan')

        # Speedup calculation: mpc.pytorch / acados
        speedup_sens = timing_mpytorch_sens / timing_ac_sens

        row_parts.extend([f"{timing_ac_sens:.3f}" if not math.isnan(timing_ac_sens) else "-",
                          f"{timing_mpytorch_sens:.2f}" if not math.isnan(timing_mpytorch_sens) else "-",
                          f"{speedup_sens:.2f}" if not math.isnan(speedup_sens) else "-"])


        # --- Append Row to Table String ---
        table_string += "| " + " | ".join(row_parts) + " |\n"

    # --- No Footer needed for Markdown ---

    # --- Write to File ---
    name = "results_table.md" if not mpc_pytorch_cuda else "results_table_cuda.md"
    with open(name, "w") as f:
        f.write(table_string)
        print(f"Written to {name}")


def analyze_constraint_activeness_in_results():
    config = PROBLEM_CONFIGS[1]
    umax, nx, nu = config
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


if __name__ == "__main__":
    # ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # num_threads = 4  # experiments in paper were run with this for the i7-8665U CPU
    # num_threads = 16  # experiments in paper were run with this for the AMD Ryzen 9 5950X 16-Core Processor
    num_threads = None  # -> detect number of threads automatically.

    # if cuda is available, use it (or set this manually to True/False)
    with_mpc_pytorch_cuda = torch.cuda.is_available()

    # run experiment
    main_experiment(with_mpc_pytorch=True, with_mpc_pytorch_cuda=with_mpc_pytorch_cuda, num_threads=num_threads)

    # evaluation
    if with_mpc_pytorch_cuda:
        evaluate_experiment_latex(mpc_pytorch_cuda=with_mpc_pytorch_cuda)
        evaluate_experiment_markdown(mpc_pytorch_cuda=with_mpc_pytorch_cuda)
    else:
        evaluate_experiment_latex()
        evaluate_experiment_markdown()

    analyze_constraint_activeness_in_results()
