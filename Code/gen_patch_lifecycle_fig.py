#!/usr/bin/env python3
"""Fig: patch-leaving is marginal-value-like AND blocking-driven (paper limit i, now measured).
(a) giving-up density -- resource remaining when a patch's occupancy collapses -- peaks well above
    zero: patches are ABANDONED before they are drained (Charnov's leave-before-empty).
(b) the fraction of departures that are BLOCKING-driven (agent still has active neighbours but
    a/k<theta) rises with theta: crowding, not just depletion, releases agents -- and tunably so.
-> theory_paper_v5/figures/fig_patch_lifecycle.pdf
"""
import os, glob
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11})

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v5", "figures")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))

# (a) giving-up-density histogram at theta=0.20
d = np.load(os.path.join(SW, "patch_rw_th0.20.npz"))
gud = d["gud"]
ax[0].hist(gud, bins=np.linspace(0, 1, 21), color="#2e7d32", alpha=0.8, edgecolor="white")
med = float(np.median(gud))
ax[0].axvline(med, color="k", ls="--", lw=1.5)
ax[0].text(med + 0.03, ax[0].get_ylim()[1] * 0.88, f"median\n{med:.2f}", fontsize=11)
ax[0].axvline(0, color="#c0392b", ls=":", lw=1.5)
ax[0].text(0.02, ax[0].get_ylim()[1] * 0.5, "drained\n(≈0)", color="#c0392b", fontsize=10)
ax[0].set_xlabel("resource remaining at departure  (giving-up density)")
ax[0].set_ylabel("number of patches")
ax[0].set_title("(a)", loc="left", fontweight="bold", fontsize=13)

# (b) blocked-release fraction vs theta
thetas, frac = [], []
for f in sorted(glob.glob(os.path.join(SW, "patch_rw_th*.npz"))):
    dd = np.load(f)
    th = float(str(f).split("th")[-1].replace(".npz", ""))
    thetas.append(th); frac.append(float(dd["blocked_release_frac"]))
o = np.argsort(thetas); thetas = np.array(thetas)[o]; frac = np.array(frac)[o]
ax[1].plot(thetas, frac, "o-", color="#1f77b4", ms=7, lw=2)
ax[1].set_ylim(-0.03, 1.0)
ax[1].set_xlabel(r"recruitment threshold $\theta$")
ax[1].set_ylabel("fraction of departures\ncaused by blocking")
ax[1].set_title("(b)", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "fig_patch_lifecycle.pdf")
fig.savefig(p, bbox_inches="tight"); print("wrote", os.path.normpath(p))
print(f"GUD median={med:.2f}; theta->blocked-release: {list(zip(thetas.round(2), frac.round(2)))}")
