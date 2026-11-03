#!/usr/bin/env python3
"""Fig 5 (fig_phasediagram) from the CONTROLLED alpha=1 grid (gen_q4_phasegrid_v15). 6x6 heatmap of
log2 harvest ratio over cover (sparse->diffuse) x src_per_clump (clustering); cells hatched where the
95% bootstrap CI includes 1 (indistinguishable from baseline). Replaces the alpha-marginal Sobol figure
so the figure matches its 'computed at alpha=1' caption."""
import os, numpy as np, pandas as pd, matplotlib
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13})
np.seterr(all="ignore")
D = os.path.join(_P.DATA, "q4_phasegrid_v15")
FIGDIR = _P.FIG
df = pd.read_csv(f"{D}/q4_phasegrid_v15.csv"); raw = np.load(f"{D}/q4_phasegrid_v15_raw.npz")
rng = np.random.default_rng(1)
nc, ns = df.ci.max() + 1, df.si.max() + 1
L2 = np.full((ns, nc), np.nan); incl1 = np.zeros((ns, nc), bool)
covers = np.full(nc, np.nan); spcs = np.full(ns, np.nan)
for r in df.itertuples():
    ci, si = int(r.ci), int(r.si); covers[ci] = r.cover; spcs[si] = r.src_per_clump
    L2[si, ci] = np.log2(r.ratio)
    Ur, Ub = raw[f"c{ci}s{si}__Urec"], raw[f"c{ci}s{si}__Ubase"]; n = len(Ur); o = np.empty(2000)
    for b in range(2000):
        s = rng.integers(0, n, n); o[b] = Ur[s].sum() / max(Ub[s].sum(), 1e-12)
    lo, hi = np.percentile(o, [2.5, 97.5]); incl1[si, ci] = (lo <= 1.0 <= hi)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
vmax = np.nanmax(np.abs(L2))
im = ax.imshow(L2, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
for si in range(ns):
    for ci in range(nc):
        if incl1[si, ci]:
            ax.add_patch(plt.Rectangle((ci - 0.5, si - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor="0.3", lw=0.0, alpha=0.5))
        ax.text(ci, si, f"{2**L2[si,ci]:.1f}", ha="center", va="center", fontsize=8,
                color="k" if abs(L2[si, ci]) < 0.6 * vmax else "w")
ax.set_xticks(range(nc)); ax.set_xticklabels([f"{c:.3g}" for c in covers], rotation=45, ha="right")
ax.set_yticks(range(ns)); ax.set_yticklabels([f"{s:.1f}" for s in spcs])
ax.set_xlabel(r"coverage (sparse $\rightarrow$ diffuse)")
ax.set_ylabel(r"sources per clump (clustering $\rightarrow$)")
ax.set_title(r"Harvest ratio $U_{\rm rec}/U_{\rm base}$  ($\alpha=1$); $>1$ = recruitment wins")
cb = fig.colorbar(im); cb.set_label(r"$\log_2$ harvest ratio")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_phasediagram.pdf"); fig.savefig(f"{FIGDIR}/fig_phasediagram.png", dpi=140)
print(f"wrote fig_phasediagram.pdf | peak {df.ratio.max():.2f} min {df.ratio.min():.2f} "
      f"wins {(df.ratio>1).sum()}/{len(df)}; {int(incl1.sum())} cells CI-includes-1")
