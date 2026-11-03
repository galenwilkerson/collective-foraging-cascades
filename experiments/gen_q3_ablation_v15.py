#!/usr/bin/env python3
"""RQ3 Fig 4 (fig_q3_ablation) on the CORRECTED engine -> v15/figures. Fresh compute, NO cache load
(the Code/ version silently re-loaded a pre-correction q3_ablation.npz and wrote to theory_paper_v6).
The coupling->motion intervention: hold everything fixed, toggle only `converge`. In the corrected engine
recruited agents ALWAYS take the short step; converge only flips the heading from parent-centroid-directed
(coupled) to random (ablated), so this is a clean DIRECTION-only ablation at fixed step size.
(a) active fraction vs tick: coupled self-limits, ablated floods toward 1.
(b) steady-state distribution of active fraction.
Config = the CANONICAL operating point (rho_r=3, theta=0.20, phi=0.05), shared with the RQ2 gate/cycle/
spectrum so active-fraction numbers are comparable across all single-point mobile figures (standardized
2026-11-01, rho_r 2.1->3). Clustered field (clump_count=6) kept, matching the paper's clustered focus.
B=96 for tighter bands/GPU.
"""
import os, sys, time
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})
sys.path.insert(0, _P.CODE)
from mobile_cascade import MobileCascade, MobileCascadeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
DATADIR = os.path.join(HERE, "sweeps")
C_COUP, C_ABL = "#2e7d32", "#c0392b"


def run(converge, B=96, N=120, T=1000, seed=0):
    c = MobileCascadeConfig(B=B, N=N, K=15, L=800.0, rho_r=3.0, phi=0.05, theta=0.20,
                            levy_alpha=1.0, clump_count=6, clump_radius=40.0, capacity=40.0,
                            consumption=1.0, step=20.0, pull=0.12, exploit_step=3.0,
                            converge=converge, device="cuda", seed=seed)
    o = MobileCascade(c).run(T)
    return 1.0 - o["exploring_fraction"].cpu().numpy()      # active fraction (B, T)


torch.cuda.reset_peak_memory_stats(); t0 = time.time()
coup, abl = run(True), run(False)                           # fresh, always
vram = torch.cuda.max_memory_allocated() / 1e9
t = np.arange(coup.shape[1]); tail = slice(200, None)
os.makedirs(DATADIR, exist_ok=True)
np.savez(os.path.join(DATADIR, "q3_ablation_v15.npz"), coup=coup, abl=abl)

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
a2.legend(loc="center right", framealpha=0.9)
a2.set_title("(b)", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout(); os.makedirs(FIGDIR, exist_ok=True)
out = os.path.join(FIGDIR, "fig_q3_ablation.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}  ({time.time()-t0:.0f}s, VRAM {vram:.2f}GB, B={coup.shape[0]})")
print(f"coupled steady mean={coup[:, tail].mean():.3f}; ablated steady mean={abl[:, tail].mean():.3f}")
