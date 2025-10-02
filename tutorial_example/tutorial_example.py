import argparse
import numpy as np
from pathlib import Path
import casadi as ca
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver, latexify_plot
import matplotlib.pyplot as plt
latexify_plot()

def export_parametric_ocp() -> AcadosOcp:

    model = AcadosModel()
    model.x = ca.SX.sym("x", 1)
    model.p_global = ca.SX.sym("p_global", 1)
    model.cost_expr_ext_cost_e = (model.x - model.p_global**2)**2
    model.name = "non_ocp"
    ocp = AcadosOcp()
    ocp.model = model

    ocp.constraints.lbx_e = np.array([-1.0])
    ocp.constraints.ubx_e = np.array([1.0])
    ocp.constraints.idxbx_e = np.array([0])

    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.N_horizon = 0

    ocp.p_global_values = np.zeros((1,))
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.qp_solver_cond_ric_alg = 0

    return ocp

def solve_and_compute_sens(p_test, tau):
    np_test = p_test.shape[0]

    ocp = export_parametric_ocp()
    ocp.solver_options.tau_min = tau
    ocp.solver_options.qp_solver_t0_init = 0
    ocp.solver_options.nlp_solver_max_iter = 2 # QP should converge in one iteration

    ocp_solver = AcadosOcpSolver(ocp, json_file="parameter_augmented_acados_ocp.json", verbose=False)

    sens_x = np.zeros(np_test)
    solution = np.zeros(np_test)

    for i, p in enumerate(p_test):
        p_val = np.array([p])

        ocp_solver.set_p_global_and_precompute_dependencies(p_val)
        status = ocp_solver.solve()
        solution[i] = ocp_solver.get(0, "x")[0]

        if status != 0:
            ocp_solver.print_statistics()
            raise Exception(f"OCP solver returned status {status} at {i}th p value {p}, {tau=}.")
            # print(f"OCP solver returned status {status} at {i}th p value {p}, {tau=}.")
            # breakpoint()

        # Calculate the policy gradient
        status = ocp_solver.setup_qp_matrices_and_factorize()
        out_dict = ocp_solver.eval_solution_sensitivity(0, "p_global", return_sens_x=True, return_sens_u=False)
        sens_x[i] = out_dict['sens_x'].item()

    return solution, sens_x

def main(args):
    p_nominal = 0.0
    delta_p = 0.002
    p_test = np.arange(p_nominal - 2, p_nominal + 2, delta_p)
    sens_list = []
    labels_list = []
    sol_list = []
    tau = 1e-6
    solution, sens_x = solve_and_compute_sens(p_test, tau)

    # Compare to numerical gradients
    sens_x_fd = np.gradient(solution, delta_p)
    test_tol = 1e-2
    median_diff = np.median(np.abs(sens_x - sens_x_fd))
    print(f"Median difference between policy gradient obtained by acados and via FD is {median_diff} should be < {test_tol}.")
    # test: check median since derivative cannot be compared at active set changes
    assert median_diff <= test_tol

    sens_list.append(sens_x)
    labels_list.append(r"$\tau = 10^{-6}$")
    sol_list.append(solution)

    tau_vals = [1e-4, 1e-3, 1e-2]
    for tau in tau_vals:
        sol_tau, sens_x_tau = solve_and_compute_sens(p_test, tau)
        sens_list.append(sens_x_tau)
        labels_list.append(r"$\tau = 10^{" + f"{int(np.log10(tau))}" + r"}$")
        # labels_list.append(r"$\tau =" + f"{tau}" + r"$")
        sol_list.append(sol_tau)

    fig_filename = args.path / f"solution_sens_non_ocp.{args.type}"
    plot_solution_sensitivities_results(p_test, sol_list, sens_list, labels_list,
                 title=None, parameter_name=r"$\theta$", fig_filename=fig_filename)
    fig_filename = args.path / f"solution_sens_non_ocp_transposed.{args.type}"
    plot_solution_sensitivities_results(p_test, sol_list, sens_list, labels_list,
                 title=None, parameter_name=r"$\theta$", fig_filename=fig_filename, horizontal_plot=True)

def plot_solution_sensitivities_results(p_test, sol_list, sens_list, labels_list, title=None, parameter_name="", fig_filename=None, horizontal_plot=False):
    p_min = p_test[0]
    p_max = p_test[-1]
    linestyles = ["--", "-.", "--", ":", "-.", ":"]

    nsub = 2
    if horizontal_plot:
        _, ax = plt.subplots(nrows=1, ncols=nsub, sharex=False, figsize=(12, 2.8))
    else:
        _, ax = plt.subplots(nrows=nsub, ncols=1, sharex=True, figsize=(6.5, 3.8))

    isub = 0
    # plot analytic solution
    ax[isub].plot([p_min, -1], [1, 1], "k-", linewidth=2, label="analytic")
    ax[isub].plot([1, p_max], [1, 1], "k-", linewidth=2)
    x_vals = np.linspace(-1, 1, 100)
    y_vals = x_vals**2
    ax[isub].plot(x_vals, y_vals, "k-", linewidth=2)

    for i, sol in enumerate(sol_list):
        ax[isub].plot(p_test, sol, label=labels_list[i], linestyle=linestyles[i])
    ax[isub].set_xlim([p_test[0], p_test[-1]])
    ax[isub].set_ylabel(r"solution $x^{\star}$")
    if title is not None:
        ax[isub].set_title(title)
    ax[isub].legend()

    isub += 1

    # plot analytic sensitivity
    ax[isub].plot([p_min, -1], [0, 0], "k-", linewidth=2, label="analytic")
    ax[isub].plot([1, p_max], [0, 0], "k-", linewidth=2)
    ax[isub].plot([-1, 1], [-2, 2], "k-", linewidth=2)

    # plot numerical sensitivities
    for i, sens_x_tau in enumerate(sens_list):
        ax[isub].plot(p_test, sens_x_tau, label=labels_list[i], color=f"C{i}", linestyle=linestyles[i])
    ax[isub].set_xlim([p_test[0], p_test[-1]])
    ax[isub].set_ylabel(r"derivative $\partial_\theta x^{\star}$")
    # ax[isub].legend(ncol=2)

    for i in range(nsub):
        ax[i].grid(True)
        if horizontal_plot:
            ax[i].set_xlabel(f"{parameter_name}")
    ax[-1].set_xlabel(f"{parameter_name}")

    plt.tight_layout()

    if fig_filename is not None:
        plt.savefig(fig_filename, dpi=args.png_dpi)
        print(f"stored figure as {fig_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default="figures", help="Where to save the plots")
    parser.add_argument("--type", type=str, default="png", help="Filetype of the plots", choices=['png', 'pdf'])
    parser.add_argument("--png_dpi", type=int, default=300, help="DPI of the png plots")

    args = parser.parse_args()

    # ensure path directory exists
    args.path.mkdir(parents=True, exist_ok=True)

    main(args)

    # to plot only analytic solution
    # plot_solution_sensitivities_results([-2, 2], [], [], [], parameter_name=r"$\theta$", fig_filename="solution_sens_non_ocp_analytic.pdf")
