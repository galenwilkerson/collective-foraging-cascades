#!/usr/bin/env python3
"""CPU replot of Fig 4 (fig_q3_ablation) from the cached q3_ablation_v15.npz -- figure polish only, no
re-simulation. Fix: panel (b) legend moved to lower right (was center right, overlaying the bars)."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
d = np.load(os.path.join(HERE, "sweeps", "q3_ablation_v15.npz"))
coup, abl = d["coup"], d["abl"]
C_COUP, C_ABL = "#2e7d32", "#c0392b"
t = np.arange(coup.shape[1]); tail = slice(200, None)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.7, 1.0]})
for A, col, lab in [(coup, C_COUP, "coupled (intact)"), (abl, C_ABL, "uncoupled")]:
    m, sem = A.mean(0), A.std(0) / np.sqrt(A.shape[0])
    a1.fill_between(t, m - sem, m + sem, color=col, alpha=0.25, linewidth=0)
    a1.plot(t, m, color=col, lw=1.9, label=lab)
a1.axhline(0.5, ls=":", color="0.4", lw=1)
a1.set_xlim(0, t.max()); a1.set_ylim(0, 1.02)
a1.set_xlabel("tick"); a1.set_ylabel("active fraction")
a1.legend(loc="lower right", framealpha=0.9)
a1.set_title("(a)", loc="left", fontweight="bold", fontsize=13)

bins = np.linspace(0, 1, 26)
for A, col, lab in [(coup, C_COUP, "coupled"), (abl, C_ABL, "uncoupled")]:
    a2.hist(A[:, tail].ravel(), bins=bins, orientation="horizontal", color=col, alpha=0.6,
            density=True, label=lab)
a2.axhline(0.5, ls=":", color="0.4", lw=1)
a2.set_ylim(0, 1.02); a2.set_xlabel("density"); a2.set_ylabel("active fraction (steady state)")
a2.legend(loc="lower right", framealpha=0.9)
a2.set_title("(b)", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout(); os.makedirs(FIGDIR, exist_ok=True)
out = os.path.join(FIGDIR, "fig_q3_ablation.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "| coupled steady=%.3f ablated steady=%.3f" % (coup[:, tail].mean(), abl[:, tail].mean()))
