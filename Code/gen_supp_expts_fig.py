#!/usr/bin/env python3
"""Two supplementary panels for v5:
(a) local cluster-size susceptibility chi=<S^2>/<S> vs connectivity -- peaks at the cascade onset
    then collapses (a third criticality signature, item ii).
(b) harvest vs Levy exponent alpha for clustered vs scattered target fields -- heavy-tailed search
    is (near-)optimal for BOTH, and clustered pays far more (item iii, empirical).
-> theory_paper_v5/figures/fig_susceptibility.pdf , fig_levy_field.pdf
"""
import os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11})

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v5", "figures")
os.makedirs(OUT, exist_ok=True)

# --- (a) susceptibility ---
d = np.load(os.path.join(SW, "local_susceptibility_rw.npz"))
z = d["z_global"]; chi = d["chi"]; cf = d["cascade_frac"]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(z, chi, "o-", color="#8e44ad", ms=6, lw=2, label=r"cluster susceptibility $\chi=\langle S^2\rangle/\langle S\rangle$")
ax.set_xlabel(r"mean degree $z$ (global)"); ax.set_ylabel(r"local susceptibility $\chi$")
pk = int(np.nanargmax(chi))
ax.axvline(z[pk], color="k", ls="--", lw=1)
ax.text(z[pk] + 0.4, chi[pk] * 0.9, f"peak at\n$z\\approx{z[pk]:.1f}$\n(cascade onset)", fontsize=11)
ax2 = ax.twinx()
ax2.plot(z, cf, "s--", color="#7f8c8d", ms=4, lw=1.2, alpha=0.8, label="cascade fraction")
ax2.set_ylabel("cascade fraction"); ax2.set_ylim(0, 1.02)
ax.legend(loc="upper right", fontsize=10.5); ax2.legend(loc="center right", fontsize=10.5)
fig.tight_layout(); p = os.path.join(OUT, "fig_susceptibility.pdf")
fig.savefig(p, bbox_inches="tight"); print("wrote", os.path.normpath(p))

# --- (b) levy x field ---
d = np.load(os.path.join(SW, "levy_field.npz"))
a = d["alphas"]; cl = d["clustered"]; sc = d["scattered"]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(a, cl, "o-", color="#2e8b57", ms=7, lw=2, label="clustered field")
ax.plot(a, sc, "s-", color="#c0392b", ms=7, lw=2, label="scattered field")
ax.set_xlabel("Lévy exponent $\\alpha$  (small $=$ heavier-tailed search)")
ax.set_ylabel("steady-state harvest")
for y, col in [(cl, "#2e8b57"), (sc, "#c0392b")]:      # mark each field's optimum
    pk = int(np.argmax(y)); ax.plot(a[pk], y[pk], "*", color=col, ms=18, mec="k", mew=0.7, zorder=5)
ax.plot([], [], "k*", ms=11, label="optimum (heavy-tailed, $\\alpha\\lesssim1$)")
ax.legend(fontsize=11)
fig.tight_layout(); p = os.path.join(OUT, "fig_levy_field.pdf")
fig.savefig(p, bbox_inches="tight"); print("wrote", os.path.normpath(p))
