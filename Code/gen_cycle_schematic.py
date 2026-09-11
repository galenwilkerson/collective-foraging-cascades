#!/usr/bin/env python3
"""Sandpile-analog schematic (left panel of the cycle figure): the explore->recruit->consume->blocked
cycle in the (network/spatial density x node activity) plane -- same plane as the empirical flow portrait
-- as a clean single clockwise loop matching the data's circulation. Four states coloured to match the
data panel; inward arrows on the density axis show the swarm is driven to the poised window from both
sides. The BTW-sandpile analogy: a driven-dissipative drive->avalanche->discharge->reset cycle
self-organised to a poised attractor, on a MOBILE lattice. -> theory_paper_v6/figures/fig_cycle_schematic.pdf
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "theory_paper_v6", "figures", "fig_cycle_schematic.pdf"))
PNG = os.path.join(os.environ.get("SCRATCH_PNG_DIR", "/tmp"), "fig_cycle_schematic.png")
plt.rcParams.update({"font.size": 13})
# soft fills / saturated edges+labels, to match the data panel without being gaudy
STY = {"explore": ("#e8e8e8", "#5f5f5f"), "recruit": ("#dbe4fb", "#1f4fd8"),
       "consume": ("#dcefda", "#1a7a1a"), "blocked": ("#f6dcdc", "#b01515")}

fig, ax = plt.subplots(figsize=(6.6, 5.0))
ax.set_xlim(0, 1); ax.set_ylim(-0.24, 1.05); ax.axis("off")

# --- poised-window band (pale), between z_c^RGG and 2/theta ---
ax.add_patch(Rectangle((0.35, 0.0), 0.33, 1.0, fc="#eef2f7", ec="none", zorder=0))
for xv in (0.35, 0.68):
    ax.plot([xv, xv], [0, 1], color="0.7", ls=":", lw=1.0, zorder=1)
ax.text(0.515, 1.0, "poised window", ha="center", va="bottom", fontsize=10, color="0.45", style="italic")
ax.text(0.35, -0.01, r"$z_c^{\rm RGG}$", ha="center", va="top", fontsize=10, color="0.5")
ax.text(0.68, -0.01, r"$2/\theta$", ha="center", va="top", fontsize=10, color="0.5")

# --- axes ---
ax.annotate("", xy=(1.0, 0.0), xytext=(0.02, 0.0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.45"), zorder=1)
ax.annotate("", xy=(0.02, 1.0), xytext=(0.02, 0.0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.45"), zorder=1)
ax.text(-0.045, 0.52, "node activity", ha="center", va="center", rotation=90, fontsize=11.5, color="0.35")

# --- states on a clockwise loop mirroring the data circulation ---
nodes = {"explore": (0.13, 0.14, "explore", (-0.06, -0.02, "right")),
         "recruit": (0.34, 0.58, "recruit", (-0.06, 0.0, "right")),
         "consume": (0.58, 0.86, "consume", (0.06, 0.0, "left")),
         "blocked": (0.86, 0.33, "blocked", (0.06, 0.0, "left"))}
for k, (x, y, lab, (lx, ly, ha)) in nodes.items():
    fc, ec = STY[k]
    ax.scatter([x], [y], s=2700, facecolor=fc, edgecolor=ec, linewidths=2.0, zorder=4)
    ax.text(x + lx, y + ly, lab, ha=ha, va="center", fontsize=12, color=ec, weight="bold", zorder=5)

def loop(a, b, rad, lab="", lpos=0.5, dy=0.0):
    (x0, y0, *_), (x1, y1, *_) = nodes[a], nodes[b]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=18, lw=2.0, color="#333", shrinkA=20, shrinkB=20, zorder=3))
    if lab:
        ax.text(x0 + lpos * (x1 - x0), y0 + lpos * (y1 - y0) + dy, lab, ha="center", va="center",
                fontsize=8.8, color="0.3", style="italic",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75), zorder=6)

loop("explore", "recruit", -0.15, "discovery\n(trigger)", 0.5, 0.02)
loop("recruit", "consume", -0.12, "avalanche", 0.5, 0.0)
loop("consume", "blocked", -0.15, "consume +\nblock", 0.5, 0.06)
loop("blocked", "explore", -0.22, "disperse\n(reset)", 0.5, -0.05)

# --- inward arrows on the density axis: driven to the window from BOTH sides ---
yb = -0.14
ax.add_patch(FancyArrowPatch((0.04, yb), (0.33, yb), arrowstyle="-|>", mutation_scale=20, lw=2.6, color="0.55", zorder=4))
ax.add_patch(FancyArrowPatch((0.98, yb), (0.70, yb), arrowstyle="-|>", mutation_scale=20, lw=2.6, color="0.55", zorder=4))
ax.text(0.18, yb + 0.04, "recruitment\naggregates", ha="center", va="bottom", fontsize=8.6, color="0.35")
ax.text(0.84, yb + 0.04, "blocking + consumption\ndisperse", ha="center", va="bottom", fontsize=8.6, color="0.35")
ax.text(0.5, -0.225, r"network / spatial density  (sparse $\to$ poised $\to$ dense)",
        ha="center", va="top", fontsize=11.5, color="0.35")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight"); fig.savefig(PNG, dpi=130, bbox_inches="tight")
print("wrote", OUT)
