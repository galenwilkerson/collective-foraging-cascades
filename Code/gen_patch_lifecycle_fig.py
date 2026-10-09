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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))

# (a) giving-up-density histogram at theta=0.20
d = np.load(os.path.join(SW, "patch_rw_th0.20.npz"))
gud = d["gud"]
ax[0].hist(gud, bins=np.linspace(0, 1, 21), color="#2e7d32", alpha=0.8, edgecolor="white")
med = float(np.median(gud))
ymax = ax[0].get_ylim()[1]
ax[0].axvline(med, color="k", ls="--", lw=1.5)
ax[0].text(med + 0.03, ymax * 0.96, f"median\n{med:.2f}", fontsize=11, ha="left", va="top")
ax[0].axvline(0, color="#c0392b", ls=":", lw=1.5)
# top-left strip is the only clear space (first bars are short); keep it narrow (two lines)
# so it clears the tall central bars near x~0.1-0.35
ax[0].text(0.02, ymax * 0.985, "drained\n(≈0)", color="#c0392b", fontsize=10,
           ha="left", va="top")
ax[0].set_xlabel("resource remaining at departure  (giving-up density)")
ax[0].set_ylabel("number of patches")
ax[0].set_title("(a)", loc="left", fontweight="bold", fontsize=13)

# (b) blocked-release fraction vs theta, with a Wilson 95% CI on the proportion
# (each point is the fraction of n departures caused by blocking -- a binomial proportion,
#  not a run average, so the uncertainty is a confidence interval on that proportion).
def wilson(p, n, z=1.96):
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half

thetas, frac, lo, hi = [], [], [], []
for f in sorted(glob.glob(os.path.join(SW, "patch_rw_th*.npz"))):
    dd = np.load(f)
    th = float(str(f).split("th")[-1].replace(".npz", ""))
    p = float(dd["blocked_release_frac"]); n = int(dd["n_patches"])
    l, u = wilson(p, n)
    thetas.append(th); frac.append(p); lo.append(l); hi.append(u)
o = np.argsort(thetas)
thetas = np.array(thetas)[o]; frac = np.array(frac)[o]
lo = np.array(lo)[o]; hi = np.array(hi)[o]
yerr = np.vstack([frac - lo, hi - frac])
ax[1].errorbar(thetas, frac, yerr=yerr, fmt="o-", color="#1f77b4", ms=7, lw=2,
               capsize=3, elinewidth=1.3)
ax[1].set_ylim(-0.05, 1.0)
ax[1].set_xlabel(r"recruitment threshold $\phi$")
ax[1].set_ylabel("fraction of departures\ncaused by blocking")
ax[1].set_title("(b)", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "fig_patch_lifecycle.pdf")
fig.savefig(p, bbox_inches="tight"); print("wrote", os.path.normpath(p))
print(f"GUD median={med:.2f}; theta->blocked-release: {list(zip(thetas.round(2), frac.round(2)))}")
