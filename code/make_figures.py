"""
make_figures.py
==============
Produce all publication figures (vector PDF + 300-dpi PNG, colourblind-safe).

Figures:
  fig_field         : representative temperature field + conductivity map
  fig_convergence   : MMS L2 convergence (rate~2) + QoI mesh convergence
  fig_parity        : parity plots for GP / GBR / MLP on the held-out test set
  fig_calibration   : GP reliability curve + standardized-residual histogram
  fig_active        : RMSE vs FE calls, active vs random (mean +/- std band)
  fig_speedup       : speed-up bar (log scale) GP & GBR vs FE solver
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from solver import solve_heat, _kappa_fn

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

# Colourblind-safe palette (Wong, 2011)
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}

rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.3, "lines.linewidth": 1.8, "font.family": "serif",
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  wrote", name)


def fig_field():
    theta = [4.0, 0.4, 14.0, 1.0]
    res = solve_heat(theta, refine=7, return_field=True)
    mesh, u = res["mesh"], res["u"]
    x, y = mesh.p
    tri = mesh.t.T
    kappa = _kappa_fn(theta[0], theta[1])(mesh.p)

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    c0 = ax[0].tripcolor(x, y, tri, kappa, shading="gouraud", cmap="viridis")
    ax[0].set_title(r"Conductivity $\kappa(x,y)$")
    fig.colorbar(c0, ax=ax[0], shrink=0.85)
    c1 = ax[1].tripcolor(x, y, tri, u, shading="gouraud", cmap="inferno")
    ax[1].tricontour(x, y, tri, u, levels=8, colors="white",
                     linewidths=0.4, alpha=0.6)
    ax[1].set_title(r"Temperature $u(x,y)$")
    fig.colorbar(c1, ax=ax[1], shrink=0.85)
    for a in ax:
        a.set_xlabel("$x$"); a.set_ylabel("$y$"); a.set_aspect("equal")
        a.grid(False)
    fig.suptitle(r"Representative solve: $C=4,\ c_x=0.4,\ Q=14,\ u_D=1$",
                 y=1.02, fontsize=10)
    save(fig, "fig_field")


def fig_convergence():
    mms = pd.read_csv(os.path.join(RESULTS, "mms_convergence.csv"))
    qoi = pd.read_csv(os.path.join(RESULTS, "qoi_mesh_convergence.csv"))
    rate = np.polyfit(np.log(mms.h), np.log(mms.l2_error), 1)[0]

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax[0].loglog(mms.h, mms.l2_error, "o-", color=CB["blue"],
                 label=f"FE (P1), slope={rate:.2f}")
    ref = mms.l2_error.iloc[0] * (mms.h / mms.h.iloc[0]) ** 2
    ax[0].loglog(mms.h, ref, "--", color=CB["black"], label=r"$O(h^2)$ ref.")
    ax[0].set_xlabel("mesh size $h$"); ax[0].set_ylabel(r"$L_2$ error")
    ax[0].set_title("Manufactured-solution convergence"); ax[0].legend()

    axr = ax[1]
    axr.semilogx(qoi.ndofs, qoi.Tavg, "s-", color=CB["green"],
                 label=r"$T_{\rm avg}$")
    axr.set_xlabel("number of DOFs"); axr.set_ylabel(r"$T_{\rm avg}$")
    axr.set_title("QoI mesh convergence")
    ax2 = axr.twinx()
    ax2.semilogx(qoi.ndofs, qoi.Tmax, "^--", color=CB["vermillion"],
                 label=r"$T_{\rm max}$")
    ax2.set_ylabel(r"$T_{\rm max}$", color=CB["vermillion"])
    ax2.grid(False)
    axr.axvline(4225, color="grey", ls=":", lw=1)
    axr.text(4225, qoi.Tavg.min(), " chosen mesh", fontsize=8, color="grey")
    lines = axr.get_lines() + ax2.get_lines()
    axr.legend(lines, [l.get_label() for l in lines], loc="center right")
    save(fig, "fig_convergence")


def fig_parity():
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.5))
    cols = {"GP": CB["blue"], "GBR": CB["orange"], "MLP": CB["green"]}
    met = pd.read_csv(os.path.join(RESULTS, "surrogate_metrics.csv")).set_index("model")
    for a, name in zip(ax, ["GP", "GBR", "MLP"]):
        d = pd.read_csv(os.path.join(RESULTS, f"parity_{name}.csv"))
        lo = min(d.y_true.min(), d.y_pred.min())
        hi = max(d.y_true.max(), d.y_pred.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.scatter(d.y_true, d.y_pred, s=14, color=cols[name], alpha=0.7,
                  edgecolor="none")
        r2 = met.loc[name, "test_R2"]; rm = met.loc[name, "test_RMSE"]
        a.set_title(f"{name}\n$R^2$={r2:.4f}, RMSE={rm:.2e}")
        a.set_xlabel(r"FE $T_{\rm avg}$"); a.set_ylabel("surrogate")
        a.set_aspect("equal", adjustable="box")
    save(fig, "fig_parity")


def fig_calibration():
    cal = pd.read_csv(os.path.join(RESULTS, "gp_calibration.csv"))
    res = pd.read_csv(os.path.join(RESULTS, "gp_residuals.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax[0].plot(cal.nominal, cal.observed, "o-", color=CB["purple"],
               label="GP")
    ax[0].set_xlabel("nominal coverage"); ax[0].set_ylabel("observed coverage")
    ax[0].set_title("GP reliability diagram"); ax[0].legend()
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    z = res.z.values
    ax[1].hist(z, bins=20, density=True, color=CB["sky"],
               edgecolor="k", alpha=0.8, label="std. residuals")
    xs = np.linspace(-4, 4, 200)
    ax[1].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "-",
               color=CB["black"], label=r"$\mathcal{N}(0,1)$")
    ax[1].set_xlabel(r"$(y-\mu)/\sigma$"); ax[1].set_ylabel("density")
    ax[1].set_title("Standardized residuals"); ax[1].legend()
    save(fig, "fig_calibration")


def fig_active():
    df = pd.read_csv(os.path.join(RESULTS, "active_learning_curves.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    cols = {"active": CB["vermillion"], "random": CB["blue"]}
    for method in ["active", "random"]:
        sub = df[df.method == method]
        g = sub.groupby("n_train")
        m = g["rmse"].mean(); s = g["rmse"].std()
        n = m.index.values
        ax[0].plot(n, m.values, "-o", color=cols[method], ms=4,
                   label=method)
        ax[0].fill_between(n, m - s, m + s, color=cols[method], alpha=0.2)
        mr = g["r2"].mean()
        ax[1].plot(n, mr.values, "-o", color=cols[method], ms=4, label=method)
    ax[0].axhline(0.005, color="grey", ls=":", label="target 0.005")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("number of FE solves"); ax[0].set_ylabel("test RMSE")
    ax[0].set_title("Active vs random sampling"); ax[0].legend()
    ax[1].set_xlabel("number of FE solves"); ax[1].set_ylabel(r"test $R^2$")
    ax[1].set_title("Test $R^2$ vs budget"); ax[1].legend(loc="lower right")
    save(fig, "fig_active")


def fig_speedup():
    with open(os.path.join(RESULTS, "speedup.json")) as fh:
        sp = json.load(fh)
    labels = ["FE solver", "GBR", "GP"]
    times_us = [sp["fe_solve_seconds"] * 1e6,
                sp["gbr_predict_seconds"] * 1e6,
                sp["gp_predict_seconds"] * 1e6]
    cols = [CB["black"], CB["orange"], CB["blue"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(labels, times_us, color=cols, edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel(r"cost per evaluation [$\mu$s]")
    ax.set_title("Evaluation cost (log scale)")
    sus = {"GBR": sp["speedup_gbr"], "GP": sp["speedup_gp"]}
    for b, lab in zip(bars[1:], ["GBR", "GP"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.4,
                f"{sus[lab]:,.0f}×", ha="center", fontsize=9)
    save(fig, "fig_speedup")


if __name__ == "__main__":
    print("Generating figures ...")
    fig_field()
    fig_convergence()
    fig_parity()
    fig_calibration()
    fig_active()
    fig_speedup()
    print("All figures written to", FIGS)
