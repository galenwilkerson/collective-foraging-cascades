#!/usr/bin/env python3
"""Redraw fig_mechanism.pdf (Fig 1a / overview panel a): the spatial coordination primitive.
Exploring agents (open) search long-range; one finds a resource (seed); a recruitment cascade
(filled) draws neighbours in; a dense clump self-limits (blocking). Large fonts, no title.
-> theory_paper_v3/figures/fig_mechanism.pdf
"""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
plt.rcParams.update({"font.size": 14})

fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.set_xlim(0, 1.4); ax.set_ylim(0, 1); ax.axis("off")


def agent(x, y, filled, r=0.021):
    ax.add_patch(Circle((x, y), r, fc=("k" if filled else "white"), ec="k", lw=1.6, zorder=3))


def arrow(x0, y0, x1, y1, color="k", lw=1.7, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                 lw=lw, color=color, ls=ls, shrinkA=3, shrinkB=3, zorder=2))


# exploring agents (left): open circles, random-direction arrows
for x, y, dx, dy in [(0.12, 0.80, 0.06, 0.06), (0.09, 0.45, -0.05, 0.05),
                     (0.19, 0.20, 0.06, -0.03), (0.30, 0.63, 0.05, 0.05)]:
    agent(x, y, False); arrow(x, y, x + dx, y + dy)
ax.text(0.15, 0.04, "exploration\n(long-range search)", ha="center", va="top", fontsize=13)

# resource (right) with a seed agent on its edge
rx, ry = 1.20, 0.55
ax.add_patch(Circle((rx, ry), 0.13, fc="none", ec="#2c5aa0", lw=2, hatch="////", zorder=1))
ax.text(rx, ry - 0.17, "resource", ha="center", va="top", fontsize=13, color="#2c5aa0")
agent(rx - 0.13, ry, True); ax.text(rx - 0.16, ry + 0.05, "seed", ha="right", va="bottom", fontsize=12)

# recruitment cascade: filled agents converging on the resource
cascade = [(0.52, 0.49), (0.66, 0.51), (0.80, 0.52), (0.93, 0.53)]
prev = None
for x, y in cascade:
    agent(x, y, True)
    if prev:
        arrow(prev[0], prev[1], x, y, color="#2c5aa0", lw=1.9)
    prev = (x, y)
arrow(cascade[-1][0], cascade[-1][1], rx - 0.14, ry, color="#2c5aa0", lw=1.9)
ax.text(0.70, 0.40, "recruitment cascade", ha="center", va="top", fontsize=13, color="#2c5aa0")

# blocking: a dense clump self-limits (red slash)
bx, by = 0.60, 0.76
for dx, dy in [(-0.025, 0.02), (0.02, 0.02), (0.0, -0.02), (0.035, -0.015), (-0.035, -0.01)]:
    agent(bx + dx, by + dy, True, r=0.018)
ax.add_patch(FancyArrowPatch((bx - 0.07, by + 0.06), (bx + 0.08, by - 0.06),
             arrowstyle="-", lw=3.2, color="#c0392b", zorder=4))
ax.text(bx, by + 0.12, "blocking\n(self-limits)", ha="center", va="bottom", fontsize=12, color="#c0392b")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v3", "figures", "fig_mechanism.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.normpath(out))
