#!/usr/bin/env python3
"""Generate fig_invertedU.pdf (Q1) from the saved inverted_u_full.csv (produced by inverted_u.py).

Cascade fraction (+/-1 std) vs observed mean degree, with the giant-component fraction on a twin
axis. Two vertical guides mark the cascade window: the RGG percolation threshold z_c^RGG ~ 4.51
(left, "percolation onset") and the empirical blocking crossover (right, "blocking onset").
No suptitle -- the LaTeX caption carries the interpretation.  ->  theory_paper_v3/figures/fig_invertedU.pdf
"""
import os, csv
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5})

Z_C_RGG = 4.51    # critical mean degree for giant-component percolation on a 2D RGG
Z_BLOCK = 7.0     # empirical blocking crossover (cascade peak / decline)

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
rows = list(csv.DictReader(open(os.path.join(SW, "inverted_u_full.csv"))))
rows.sort(key=lambda r: float(r["z_obs"]))
z = np.array([float(r["z_obs"]) for r in rows])
cm = np.array([float(r["cascade_mean"]) for r in rows])
cs = np.array([float(r["cascade_std"]) for r in rows])
gm = np.array([float(r["gcc_mean"]) for r in rows])

fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
l1, = ax1.plot(z, cm, color="C0", label="cascade fraction")
l2 = ax1.fill_between(z, np.clip(cm - cs, 0, 1), np.clip(cm + cs, 0, 1), color="C0", alpha=0.25,
                      linewidth=0, label=r"$\pm1$ std")
ax1.set_xlabel("Mean degree (observed)")
ax1.set_ylabel("Cascade fraction")
ax1.set_ylim(0.0, 1.28)                         # headroom above 1.0 for the onset labels
ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax2 = ax1.twinx()
l3, = ax2.plot(z, gm, "--", color="C0", label="GCC fraction")
ax2.set_ylabel("Giant component fraction")
ax2.set_ylim(0.0, 1.28)                         # keep twin axis on the same scale as ax1
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# --- left guide: percolation onset (z_c^RGG).  Label sits ABOVE the axes (arrow points down to
#     the line) so it never overlaps the curves; the z_c^RGG name sits low, inside, beside the line.
ax1.axvline(Z_C_RGG, color="k", ls=":", lw=1.5)
ax1.annotate("percolation\nonset", xy=(Z_C_RGG, 1.05), xytext=(Z_C_RGG - 1.9, 1.20),
             ha="center", va="center", fontsize=11, annotation_clip=False,
             arrowprops=dict(arrowstyle="-|>", lw=1.3, color="k"))
ax1.text(Z_C_RGG + 0.15, 0.03, r"$z_c^{\mathrm{RGG}}$", va="bottom", ha="left", fontsize=12)

# --- right guide: blocking onset
ax1.axvline(Z_BLOCK, color="gray", ls="--", lw=1.5)
ax1.annotate("blocking\nonset", xy=(Z_BLOCK, 1.05), xytext=(Z_BLOCK + 2.0, 1.20),
             ha="center", va="center", fontsize=11, annotation_clip=False,
             arrowprops=dict(arrowstyle="-|>", lw=1.3, color="gray"))

ax1.legend(handles=[l1, l2, l3], loc="center right", bbox_to_anchor=(1.0, 0.62),
           fontsize=10.5, framealpha=0.9)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v3", "figures", "fig_invertedU.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.normpath(out))
print(f"peak cascade {cm.max():.3f} at z={z[cm.argmax()]:.2f}")
