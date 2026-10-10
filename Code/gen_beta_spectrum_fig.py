#!/usr/bin/env python3
"""SI fig: the resource-field spectrum as a maximum-entropy (Gibbs) point process.

A point pattern with pairwise interaction, as the max-ent distribution given a mean clumping -- P ~ e^{-E},
E = a * sum_{i<j} K(r_ij), K a short-range kernel. The sign of the interaction sweeps the whole spectrum:
  a>0 (repulsive) -> REGULAR (needs a mechanism; the exception),
  a=0 (independent) -> CSR (a knife-edge; patchy only by fluctuation = the weak limit),
  a<0 (attractive) -> CLUSTERED (the generic case once aggregation is admitted).
Top row: sample patterns. Bottom: pair-correlation g(r) (flat=CSR, peaked=clustered, dip=regular).
Pure numpy Metropolis, no simulation of the swarm. -> figures/fig_beta_spectrum.pdf
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "fig_beta_spectrum.pdf")
rng = np.random.default_rng(1)
N, SIG, RMIN = 260, 0.055, 0.012   # points; interaction range; hard core (prevents collapse)


def minimg(d):
    return d - np.round(d)


def energy_point(pts, i, p, a):
    d = minimg(pts - p); r2 = (d ** 2).sum(1)
    k = np.exp(-r2 / SIG ** 2); k[i] = 0.0
    return a * k.sum()


def too_close(pts, i, p):
    d = minimg(pts - p); r2 = (d ** 2).sum(1); r2[i] = 1.0
    return (r2 < RMIN ** 2).any()


def sample(a, sweeps=450):
    pts = rng.random((N, 2))
    for _ in range(sweeps):
        for i in rng.permutation(N):
            p_new = rng.random(2)
            if too_close(pts, i, p_new):
                continue
            dE = energy_point(pts, i, p_new, a) - energy_point(pts, i, pts[i], a)
            if dE <= 0 or rng.random() < np.exp(-dE):
                pts[i] = p_new
    return pts


def gofr(pts, nb=24, rmax=0.30):
    d = pts[:, None, :] - pts[None, :, :]
    r = np.sqrt((minimg(d) ** 2).sum(2))
    r = r[np.triu_indices(N, 1)]
    edges = np.linspace(1e-3, rmax, nb + 1); rc = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(r, bins=edges)
    rho = N / 1.0
    ideal = rho * N * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / 2.0   # expected pairs per shell (CSR)
    return rc, h / np.maximum(ideal, 1e-9)


cases = [("repulsive $(\\beta<0)$:\nregular", +2.5, "#1a7a1a"),
         ("independent $(\\beta=0)$:\nCSR", 0.0, "#555555"),
         ("attractive $(\\beta>0)$:\nclustered", -0.95, "#c0392b")]
fig, ax = plt.subplots(2, 3, figsize=(9.0, 5.6), gridspec_kw={"height_ratios": [1.25, 1.0]})
gr = {}
for j, (lab, a, col) in enumerate(cases):
    pts = sample(a, sweeps=(40 if a < 0 else 450))   # arrest coarsening for the clustered case
    ax[0, j].plot(pts[:, 0], pts[:, 1], "o", ms=3.0, color=col, mec="none")
    ax[0, j].set_xlim(0, 1); ax[0, j].set_ylim(0, 1); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    ax[0, j].set_title(lab, fontsize=10)
    ax[0, j].set_aspect("equal")
    gr[j] = (gofr(pts), col, lab.split(":")[0].replace("\n", " "))
# g(r) panels share bottom row: overlay all three in the middle-bottom, hide the other two
for j in range(3):
    ax[1, j].axhline(1.0, color="0.6", ls=":", lw=1)
    (rc, g), col, _ = gr[j]
    ax[1, j].plot(rc, g, "-", color=col, lw=1.8)
    ax[1, j].set_xlabel("$r$"); ax[1, j].set_ylim(0, max(2.2, g.max() * 1.1))
    ax[1, j].set_title("pair correlation $g(r)$", fontsize=9)
    if j == 0:
        ax[1, j].set_ylabel("$g(r)$")
    ax[1, j].text(0.95, 0.9, "$g{>}1$: clustered\n$g{<}1$: regular", transform=ax[1, j].transAxes,
                  ha="right", va="top", fontsize=7, color="0.4") if j == 1 else None
fig.suptitle("Resource field as a maximum-entropy (Gibbs) point process: one interaction sign sweeps the spectrum",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); print("wrote", os.path.normpath(OUT))
for j in range(3):
    (rc, g), _, name = gr[j]
    print(f"{name:12} g(r) at smallest r = {g[0]:.2f}  (clustered>1, CSR~1, regular<1)")
