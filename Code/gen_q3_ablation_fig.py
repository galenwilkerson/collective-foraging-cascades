#!/usr/bin/env python3
"""Q3 figure: the recruitment->motion coupling CAUSES self-limitation.

Controlled intervention: hold every parameter fixed and toggle only the coupling.
(a) active fraction vs time -- the coupled swarm self-limits to ~50%, the ablated swarm
    (activation without convergence) saturates toward ~100%.
(b) steady-state distribution of active fraction -- coupled peaked near 0.5, ablated piled near 1.0.
active fraction = 1 - exploring_fraction.  CPU.  -> theory_paper_v6/figures/fig_q3_ablation.pdf
"""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mobile_cascade import MobileCascade, MobileCascadeConfig

C_COUP, C_ABL = "#2e7d32", "#c0392b"   # coupled (green), ablated (red)


def run(converge, T=1000, seed=0):
    c = MobileCascadeConfig(B=24, N=120, K=15, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                            levy_alpha=1.0, clump_count=6, clump_radius=40.0, capacity=40.0,
                            consumption=1.0, step=20.0, pull=0.12, converge=converge,
                            device="cpu", seed=seed)
    o = MobileCascade(c).run(T)
    return 1.0 - o["exploring_fraction"].cpu().numpy()   # active fraction, (B, T)


CACHE = os.path.join(HERE, "sweeps", "q3_ablation.npz")
if os.path.exists(CACHE):
    d = np.load(CACHE); coup, abl = d["coup"], d["abl"]
else:
    coup, abl = run(True), run(False)
    np.savez(CACHE, coup=coup, abl=abl)
t = np.arange(coup.shape[1])
tail = slice(200, None)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.7, 1.0]})

# (a) time series: ensemble mean with a tight SEM band (individual runs cycle; the mean self-limits)
for A, col, lab in [(coup, C_COUP, "coupled (intact)"), (abl, C_ABL, "ablated (coupling off)")]:
    m, sem = A.mean(0), A.std(0) / np.sqrt(A.shape[0])
    a1.fill_between(t, m - sem, m + sem, color=col, alpha=0.25, linewidth=0)
    a1.plot(t, m, color=col, lw=1.9, label=lab)
a1.axhline(0.5, ls=":", color="0.4", lw=1)
a1.set_xlim(0, t.max()); a1.set_ylim(0, 1.02)
a1.set_xlabel("tick"); a1.set_ylabel("active fraction")
a1.legend(loc="lower right", framealpha=0.9)
a1.set_title("(a)", loc="left", fontweight="bold", fontsize=13)

# (b) steady-state distribution (t>200, all batches)
bins = np.linspace(0, 1, 26)
for A, col, lab in [(coup, C_COUP, "coupled"), (abl, C_ABL, "ablated")]:
    a2.hist(A[:, tail].ravel(), bins=bins, orientation="horizontal", color=col, alpha=0.6,
            density=True, label=lab)
a2.axhline(0.5, ls=":", color="0.4", lw=1)
a2.set_ylim(0, 1.02); a2.set_xlabel("density"); a2.set_ylabel("active fraction (steady state)")
a2.legend(loc="center right", framealpha=0.9)
a2.set_title("(b)", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout()
out = os.path.join(HERE, "..", "theory_paper_v6", "figures", "fig_q3_ablation.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.normpath(out))
print(f"coupled steady mean={coup[:, tail].mean():.3f}; ablated steady mean={abl[:, tail].mean():.3f}")
