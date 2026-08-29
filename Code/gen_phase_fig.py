#!/usr/bin/env python3
"""Redraw fig_phase.pdf (Fig 8): behavioural state-flow of the coordination primitive, in the
network-density x node-activity plane. Clean schematic, large fonts, no title (caption carries it).
-> theory_paper_v3/figures/fig_phase.pdf
"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
plt.rcParams.update({"font.size": 14})

fig, ax = plt.subplots(figsize=(6.8, 5.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# nodes: (x = density, y = activity)
nodes = {
    "explore":  (0.16, 0.20, "exploration\n(random walk)"),
    "recruit":  (0.52, 0.55, "activation /\nrecruitment"),
    "consume":  (0.52, 0.88, "consuming"),
    "dense":    (0.86, 0.20, "dense\n(blocking)"),
}
R = 0.055
for k, (x, y, lab) in nodes.items():
    ax.add_patch(Circle((x, y), R, fill=True, fc="#eef3f8", ec="k", lw=2, zorder=3))
    if k == "consume":                       # top node: label to the right, clear of the incoming arrow
        ax.text(x + R + 0.03, y, lab, ha="left", va="center", fontsize=13, zorder=4)
    else:
        ax.text(x, y - R - 0.02, lab, ha="center", va="top", fontsize=13, zorder=4)


def arrow(a, b, rad=0.0, ls="-", lw=2.2, color="k"):
    (x0, y0, _), (x1, y1, _) = nodes[a], nodes[b]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=22, lw=lw, color=color, ls=ls,
                 shrinkA=24, shrinkB=24, zorder=2))

# cycle: explore -> recruit -> consume -> (blocking releases) -> explore
arrow("explore", "recruit", rad=0.15)
arrow("recruit", "consume", rad=0.15)
arrow("consume", "explore", rad=0.35)                 # released after depletion
arrow("recruit", "dense", rad=-0.15)                  # crowding
arrow("dense", "explore", rad=0.25, ls="--")          # blocking releases back to search
ax.text(0.70, 0.40, "blocking\nreleases", ha="center", va="center", fontsize=12, color="#c0392b")

# axes as labelled directions
ax.annotate("", xy=(0.98, 0.03), xytext=(0.05, 0.03),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="0.4"))
ax.annotate("", xy=(0.03, 0.98), xytext=(0.03, 0.05),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="0.4"))
ax.text(0.5, -0.02, r"network / spatial density  (low $\to$ critical $\to$ dense)",
        ha="center", va="top", fontsize=13, color="0.3")
ax.text(-0.03, 0.5, r"node activity  (explore $\to$ recruit $\to$ consume)",
        ha="center", va="center", rotation=90, fontsize=13, color="0.3")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v3", "figures", "fig_phase.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.normpath(out))
