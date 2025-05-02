import argparse
import os
import casadi as ca
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from acados_template import latexify_plot
latexify_plot()

x = ca.SX.sym("x", 1)
p_global = ca.SX.sym("p_global", 1)

f = (x - 1) * (x + 1) * x * x - p_global * x

lbx = -0.75
ubx = 0.75

def create_ipopt_solver():
    opts = {'ipopt.print_level':0, 'print_time':0}
    nlp = {"x": x, "f": f, "p": p_global}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return solver

def plot_minima(args):
    solver = create_ipopt_solver()

    np_test = 1000
    p_test = np.linspace(-1, 1, np_test)

    init_vals = [-1, 0, 1]
    n_init_vals = len(init_vals)
    x_sol = np.zeros((n_init_vals, np_test))
    f_sol = np.zeros((n_init_vals, np_test))

    for i, p in enumerate(p_test):
        for j, x0 in enumerate(init_vals):
            sol = solver(x0=x0, p=p, lbx=lbx, ubx=ubx)
            x_sol[j, i] = sol['x']
            f_sol[j, i] = sol['f']
            print(f"solutions {x_sol[:, i]}")

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 3), sharex=True)
    linestyles = ['-', '--', ':']
    for j in range(n_init_vals):
        axs[0].plot(p_test, x_sol[j, :], linestyle=linestyles[j], label="$x_{\mathrm{init}}="+f"{init_vals[j]}$")
        axs[1].plot(p_test, f_sol[j, :], linestyle=linestyles[j], label="$x_{\mathrm{init}}="+f"{init_vals[j]}$")
    axs[1].set_xlabel(r"$\theta$")
    axs[0].set_ylabel("$x^\star$")
    axs[1].set_ylabel("$f(x^\star)$")
    axs[0].grid()
    axs[1].grid()
    axs[0].set_xlim([p_test[0], p_test[-1]])
    axs[0].legend(handlelength=0.8)
    plt.tight_layout()
    fig_filename = args.path / f"jump_nlp_numerical_sol.{args.type}"
    plt.savefig(fig_filename, dpi=args.png_dpi)
    print(f"saved figure in {fig_filename}")


def plot_hessian(args):
    solver = create_ipopt_solver()

    hess = ca.jacobian(ca.jacobian(f, x), x)
    hess_fun = ca.Function("hess_fun", [x, p_global], [hess])

    np_test = 10000
    p_test = np.linspace(-1, 1, np_test)
    hess_vals = np.zeros_like(p_test)
    for i, p in enumerate(p_test):
        sol = solver(x0=-1.0, p=p, lbx=lbx, ubx=ubx)
        hess_vals[i] = hess_fun(sol['x'], p)

    plt.figure(figsize=(6.5, 3))
    plt.plot(p_test, hess_vals)
    plt.grid()
    plt.ylabel(r"Hessian $\nabla^2_x f(x^\star, \theta)$")
    plt.xlabel(r"$\theta$")
    plt.tight_layout()
    fig_filename = args.path / f"jump_nlp_hess.{args.type}"
    plt.savefig(fig_filename, dpi=args.png_dpi)
    print(f"saved figure in {fig_filename}")



def plot_curves(args):
    f_fun = ca.Function("f_fun", [x, p_global], [f])

    n_curves = 9
    p_test = np.linspace(-0.8, 0.8, n_curves)
    x_vals = np.linspace(lbx, ubx, 100)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_curves))
    plt.figure(figsize=(6.5, 3))
    for i, p in enumerate(p_test):
        curve_vals = np.zeros_like(x_vals)
        for j, x_val in enumerate(x_vals):
            curve_vals[j] = f_fun(x_val, p)
        plt.plot(x_vals, curve_vals, label=f"$\\theta={p:.1f}$", color=colors[i])
    plt.xlabel("$x$")
    plt.xlim([x_vals[0], x_vals[-1]])
    plt.grid()
    plt.ylabel("$f(x, \\theta)$")
    plt.ylim([-1, 0.35])
    plt.tight_layout()
    plt.legend(ncol=3, handlelength=0.8, columnspacing=1.0)
    fig_filename = args.path / f"jump_nlp_curves.{args.type}"
    plt.savefig(fig_filename, dpi=args.png_dpi)
    print(f"saved figure in {fig_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default="figures", help="Where to save the plots")
    parser.add_argument("--type", type=str, default="png", help="Filetype of the plots", choices=['png', 'pdf'])
    parser.add_argument("--png_dpi", type=int, default=300, help="DPI of the png plots")

    args = parser.parse_args()

    # ensure path directory exists
    args.path.mkdir(parents=True, exist_ok=True)

    plot_hessian(args)
    plot_minima(args)
    plot_curves(args)
    plt.show()