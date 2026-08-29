#!/usr/bin/env python3
"""Figures for the threshold-distribution sweep (sweeps/threshold_{gaussian,uniform}.npz)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v3", "figures")
plt.rcParams.update({"font.size": 13, "axes.titlesize": 12.5, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
                     "figure.dpi": 150})


def heat(ax, Z, x, y, title, cmap="viridis", ctr=None):
    ext = [0, len(x), 0, len(y)]
    vmax = None
    if ctr is not None:
        m = np.nanmax(np.abs(Z - ctr)); im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
            extent=ext, vmin=ctr - m, vmax=ctr + m)
    else:
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, extent=ext)
    ax.set_xticks(np.arange(len(x)) + 0.5); ax.set_xticklabels([f"{v:.2f}" for v in x], fontsize=8)
    ax.set_yticks(np.arange(len(y)) + 0.5); ax.set_yticklabels([f"{v:.2f}" for v in y], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    for i in range(len(y)):
        for j in range(len(x)):
            ax.text(j + 0.5, i + 0.5, f"{Z[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig_gaussian():
    F = np.load(os.path.join(SW, "threshold_gaussian.npz"))
    th, sp = F["thetas"], F["spreads"]          # rows=theta, cols=spread
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    heat(ax[0, 0], F["eff"], sp, th, "Efficiency  (harvest / cost)", "RdBu_r")
    heat(ax[0, 1], F["explore_frac"], sp, th, "Exploring fraction  (explore/exploit balance)", "cividis")
    heat(ax[1, 0], F["casc_frac"], sp, th, "Cascade (recruited) fraction", "magma")
    heat(ax[1, 1], F["blocked_frac"], sp, th, "BLOCKED fraction (suppressed by a/k < θ)", "inferno")
    for a in ax.flat:
        a.set_xlabel("threshold spread  σ"); a.set_ylabel("threshold mean  θ")
    fig.suptitle("Gaussian threshold heterogeneity — sweep over (mean θ, spread σ)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "fig_threshold_gaussian.pdf"); fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def fig_uniform():
    F = np.load(os.path.join(SW, "threshold_uniform.npz"))
    ctr, w = F["thetas"], F["spreads"]          # rows=center, cols=half-width
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, c in enumerate(ctr):
        ax[0].plot(w, F["eff"][i], "-o", label=f"θ={c:.2f}")
        ax[1].plot(w, F["explore_frac"][i], "-o", label=f"θ={c:.2f}")
        ax[2].plot(w, F["casc_frac"][i], "-o", label=f"θ={c:.2f}")
    for a, t in zip(ax, ["Efficiency", "Exploring fraction", "Cascade fraction"]):
        a.set_xlabel("uniform half-width  w"); a.set_title(t, fontsize=12.5); a.legend(fontsize=8, frameon=False)
    fig.suptitle("Uniform threshold heterogeneity — U[θ−w, θ+w]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "fig_threshold_uniform.pdf"); fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


def fig_blocking():
    """Blocking measured directly: blocked fraction and the recruited inverted-U vs connectivity.
    (Only the depletion-ON runs are shown; the depletion-OFF ablation is confounded -- with infinite
    sources agents park on them permanently and ~85% become SEEDS, so it tests nothing.)"""
    F = np.load(os.path.join(SW, "blocking_test.npz"))
    z, ths = F["z"], F["thetas"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for th in ths:
        tag = f"th{th:.2f}_dep"
        col = "#d94f3d" if th < 0.25 else "#5b8def"
        ax[0].plot(z, F[f"{tag}_blocked_frac"], "-o", color=col, ms=4, label=f"θ={th:.2f}")
        ax[1].plot(z, F[f"{tag}_casc_frac"], "-o", color=col, ms=4, label=f"θ={th:.2f}")
        for a in ax:                                   # blocking boundary for this theta
            a.axvline(2 / th, ls=":", color=col, lw=1.2, alpha=0.7)
    ax[0].set_ylabel("blocked fraction")
    ax[1].set_ylabel("recruited (cascade) fraction")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("mean degree z   (dotted = blocking boundary 2/θ)")
        a.legend(fontsize=11, frameon=False)
    ax[0].set_title("(a)", loc="left", fontweight="bold", fontsize=13)
    ax[1].set_title("(b)", loc="left", fontweight="bold", fontsize=13)
    # (suptitle omitted; the LaTeX caption carries the interpretation)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_blocking_vs_depletion.pdf"); fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    if os.path.exists(os.path.join(SW, "threshold_gaussian.npz")):
        fig_gaussian()
    if os.path.exists(os.path.join(SW, "threshold_uniform.npz")):
        fig_uniform()
    if os.path.exists(os.path.join(SW, "blocking_test.npz")):
        fig_blocking()
