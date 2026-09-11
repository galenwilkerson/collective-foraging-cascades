#!/usr/bin/env python3
"""Inverted-U cascade window with the self-tuning restoring arrows drawn as a strip BELOW the axis
(the sign of dz/dt: aggregate below the window, disperse above -- the 1-D shadow of the cycle's z-flow).
Reads Code/sweeps/inverted_u_full.csv; writes fig_invertedU_selftune.pdf into the v11 paper and the
Complex Networks abstract figure folders."""
import os, csv
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})

Z_C_RGG = 4.51; PHI = 0.15; Z_BLOCK = 2.0 / PHI; ARR = "#d1495b"
HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "sweeps")
OUTS = [os.path.join(HERE, "..", "theory_paper_v6", "figures")]

rows = list(csv.DictReader(open(os.path.join(SW, "inverted_u_full.csv")))); rows.sort(key=lambda r: float(r["z_obs"]))
z = np.array([float(r["z_obs"]) for r in rows]); cm = np.array([float(r["cascade_mean"]) for r in rows])
cs = np.array([float(r["cascade_std"]) for r in rows]); gm = np.array([float(r["gcc_mean"]) for r in rows])

fig = plt.figure(figsize=(6.6, 4.7))
gs = GridSpec(2, 1, height_ratios=[7, 1.1], hspace=0.05)
ax = fig.add_subplot(gs[0]); axp = fig.add_subplot(gs[1], sharex=ax)

ax.plot(z, cm, color="C0", lw=2)
ax.fill_between(z, np.clip(cm - cs, 0, 1), np.clip(cm + cs, 0, 1), color="C0", alpha=0.22, lw=0)
ax.set_ylabel("Cascade fraction"); ax.set_ylim(0, 1.22); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax2 = ax.twinx(); ax2.plot(z, gm, "--", color="C0", lw=1.3, alpha=0.8)
ax2.set_ylabel("Giant component fraction"); ax2.set_ylim(0, 1.22); ax2.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.axvspan(Z_C_RGG, Z_BLOCK, color="0.55", alpha=0.16, zorder=0)
ax.axvline(Z_C_RGG, color="k", ls=":", lw=1.4); ax.axvline(Z_BLOCK, color="gray", ls="--", lw=1.4)
ax.text(Z_C_RGG - 0.15, 0.08, r"$z_c^{\mathrm{RGG}}$", va="bottom", ha="right", fontsize=11)
ax.text(Z_BLOCK + 0.15, 0.08, r"$2/\varphi$", va="bottom", ha="left", fontsize=11, color="0.35")
plt.setp(ax.get_xticklabels(), visible=False)

# self-tuning arrows below the axis
axp.axvspan(Z_C_RGG, Z_BLOCK, color="0.55", alpha=0.16, zorder=0)
la0, la1 = 1.8, Z_C_RGG + 0.5; ra0, ra1 = 17.0, Z_BLOCK - 0.5
axp.annotate("", xy=(la1, 0.42), xytext=(la0, 0.42), arrowprops=dict(arrowstyle="-|>", lw=3.2, color=ARR))
axp.annotate("", xy=(ra1, 0.42), xytext=(ra0, 0.42), arrowprops=dict(arrowstyle="-|>", lw=3.2, color=ARR))
axp.text((la0 + la1) / 2, 0.92, "aggregate", ha="center", va="bottom", fontsize=8.5, color="0.3")
axp.text((ra0 + ra1) / 2, 0.92, "disperse", ha="center", va="bottom", fontsize=8.5, color="0.3")
axp.set_ylim(0, 1.0); axp.set_yticks([]); axp.set_xlim(0, 20); axp.set_xlabel("Mean degree (observed)")
for sp in ("top", "left", "right"):
    axp.spines[sp].set_visible(False)

for d in OUTS:
    p = os.path.normpath(os.path.join(d, "fig_invertedU_selftune.pdf"))
    fig.savefig(p, bbox_inches="tight"); print("wrote", p)
