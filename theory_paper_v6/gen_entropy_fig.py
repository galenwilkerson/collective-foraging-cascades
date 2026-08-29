#!/usr/bin/env python3
"""
Figure for the 'entropy of target configurations' subsection: sparse/clustered target
configurations grossly dominate homogeneous ones.

Exact density of states for M distinguishable targets in K=M cells. The number of
configurations with exactly e empty cells is
    Omega(e) = C(M,e) * (M-e)! * S(M, M-e)          [S = Stirling 2nd kind]
with sum_e Omega(e) = M^M and Omega(0) = M! (the perfectly even / homogeneous state).
We plot ln[Omega(e)/Omega(0)] -- the log-excess of configurations over the homogeneous
state -- vs the clumping order parameter e/M. It rises from 0 (homogeneous) to ~M at the
typical (Poisson) value e/M ~ 1-1/e ~ 0.37: clustered configs outnumber homogeneous by e^M.

No sampling; exact big-integer combinatorics + a safe log for large ints.
"""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def stirling2_row(M):
    """All S(M, j), j=0..M, as exact big integers (DP)."""
    S = [[0] * (M + 1) for _ in range(M + 1)]
    S[0][0] = 1
    for m in range(1, M + 1):
        for j in range(1, m + 1):
            S[m][j] = j * S[m - 1][j] + S[m - 1][j - 1]
    return S[M]


def log_bigint(n):
    if n <= 0:
        return -math.inf
    b = n.bit_length()
    if b <= 1000:
        return math.log(float(n))
    shift = b - 900
    return math.log(float(n >> shift)) + shift * math.log(2.0)


def log_omega_over_even(M):
    """ln[Omega(e)/Omega(0)] for e=0..M-1."""
    S = stirling2_row(M)
    logfact = [0.0] * (M + 1)
    for i in range(2, M + 1):
        logfact[i] = logfact[i - 1] + math.log(i)
    # ln Omega(e) = ln C(M,e) + ln (M-e)! + ln S(M, M-e)
    lnOmega = np.array([
        (logfact[M] - logfact[e] - logfact[M - e]) + logfact[M - e] + log_bigint(S[M - e])
        for e in range(M)
    ])
    return np.arange(M) / M, lnOmega    # raw ln Omega(e); normalize by ln Z = M ln M downstream


def occupancy_grid(ax):
    """One uniform-random drop of M=100 targets into K=100 cells, shown as dots jittered
    within their cells: clumped cells stack 2-4 dots, ~37% of cells stay empty (Poisson-1)."""
    side = 10; M = side * side
    rng = np.random.default_rng(3)
    cells = rng.integers(0, M, M)                         # each target -> a random cell
    r, c = cells // side, cells % side
    x = c + 0.5 + rng.uniform(-0.27, 0.27, M)
    y = r + 0.5 + rng.uniform(-0.27, 0.27, M)
    ax.scatter(x, y, s=40, color="#d9631e", edgecolor="#5a2408", linewidth=0.4, zorder=3)
    ax.set_xlim(0, side); ax.set_ylim(0, side)
    ax.set_xticks(range(side + 1)); ax.set_yticks(range(side + 1))
    ax.grid(True, color="0.8", lw=0.6); ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(length=0); ax.invert_yaxis()
    ax.set_xlabel("clumps + empties, not evenness", fontsize=9)


def main():
    fig = plt.figure(figsize=(9.8, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.4], wspace=0.34)
    ax_grid = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1])

    occupancy_grid(ax_grid)

    for M, c in [(40, "#e07b39"), (100, "#8b2d8b")]:
        x, lnO = log_omega_over_even(M)
        lnP = lnO - M * math.log(M)                     # P(e) = Omega(e)/Z,  Z = M^M (total arrangements)
        a1.plot(x, np.exp(lnP), "-", color=c, lw=2, label=f"M={M}")
        a1.fill_between(x, np.exp(lnP), color=c, alpha=0.15)
    # LINEAR probability -> all mass is heterogeneous, ~0 at homogeneous
    a1.axvline(1 / math.e, ls=":", color="k", lw=1)
    a1.set_xlim(0, 0.7)
    a1.annotate("homogeneous\n(even): $P\\approx0$", xy=(0, 0), xytext=(0.015, 0.02),
                fontsize=9, va="bottom", arrowprops=dict(arrowstyle="->", lw=1))
    a1.annotate("all mass here\n(heterogeneous)", xy=(0.37, 0.135), xytext=(0.47, 0.09),
                fontsize=9, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", lw=1))
    a1.set_xlabel("empty-cell fraction $e/M$   (0 = even $\\to$ clustered)")
    a1.set_ylabel("probability  $P(e)$")
    a1.legend(frameon=False, fontsize=9)
    # both panels fill the same row height, so equal transAxes y => aligned panel labels
    ax_grid.text(0.0, 1.03, "(a)", transform=ax_grid.transAxes, fontsize=13, fontweight="bold", va="bottom")
    a1.text(-0.14, 1.03, "(b)", transform=a1.transAxes, fontsize=13, fontweight="bold", va="bottom")
    out = os.path.join(HERE, "figures", "fig_entropy.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
