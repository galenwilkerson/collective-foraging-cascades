#!/usr/bin/env python3
"""
Aggregate the corrected Q4 Sobol cloud -> RQ4 deliverables (v15). Reads all q4_sobol_v15_off*.csv shards
(+ _raw.npz per-replica for bootstrap) and produces, from the ONE sample:
  (1) fig_phasediagram.pdf : harvest ratio over (cover x src_per_clump) -- the Fig 5 phase diagram, binned.
  (2) fig_q4_marginals.pdf : harvest ratio & movement-per-harvest vs cover, src_per_clump, phi, alpha, z.
  (3) per-axis IMPORTANCE ranking (tercile swing of the pooled harvest ratio) -> the "strongest driver".
  (4) bootstrap 95% CI on the headline sparse-clustered vs diffuse harvest ratios.
Harvest ratio per bin = pooled Sum(U_rec)/Sum(U_base) (robust to per-point U_base->0). No target assumed.

Usage: Q4_CSV_GLOB="…/sweeps/q4_sobol_v15_off*.csv" python3 q4_aggregate.py
"""
import os, sys, glob
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(_P.DATA, "q4_sobol_v15"); FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
GLOB = os.environ.get("Q4_CSV_GLOB", os.path.join(DATADIR, "q4_sobol_v15_off*.csv"))


def pooled(g):
    return float(g.sumU_rec.sum() / max(g.sumU_base.sum(), 1e-12))


def binned_ratio(df, col, nb=5, log=True):
    x = np.log10(df[col].clip(1e-9)) if log else df[col]
    edges = np.linspace(x.min(), x.max(), nb + 1)
    cen, rat, mv = [], [], []
    for i in range(nb):
        m = (x >= edges[i]) & (x <= edges[i + 1]) if i == nb - 1 else (x >= edges[i]) & (x < edges[i + 1])
        g = df[m]
        if len(g) < 3:
            continue
        c = 10 ** (0.5 * (edges[i] + edges[i + 1])) if log else 0.5 * (edges[i] + edges[i + 1])
        cen.append(c); rat.append(pooled(g))
        mv.append((g.sumD_rec.sum() / max(g.sumU_rec.sum(), 1e-12)) /
                  max(g.sumD_base.sum() / max(g.sumU_base.sum(), 1e-12), 1e-12))
    return np.array(cen), np.array(rat), np.array(mv)


def main():
    files = sorted(glob.glob(GLOB))
    if not files:
        print(f"no shards match {GLOB}"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("point")
    print(f"{len(df)} Sobol points from {len(files)} shard(s)")
    print(f"CONTROL baseline cascade mean={df.cascade_base.mean():.3f} (recruit {df.cascade_rec.mean():.1f})")

    # (1) phase diagram: cover x src_per_clump
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    xc = np.log10(df.cover); ys = np.log10(df.src_per_clump)
    xe = np.linspace(xc.min(), xc.max(), 6); ye = np.linspace(ys.min(), ys.max(), 6)
    Z = np.full((5, 5), np.nan)
    for i in range(5):
        for k in range(5):
            m = (xc >= xe[i]) & (xc < xe[i + 1] + (1e-9 if i == 4 else 0)) & \
                (ys >= ye[k]) & (ys < ye[k + 1] + (1e-9 if k == 4 else 0))
            if m.sum() >= 3:
                Z[k, i] = np.log2(pooled(df[m]))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(np.abs(Z)),
                   vmax=np.nanmax(np.abs(Z)), extent=[xe[0], xe[-1], ye[0], ye[-1]])
    ax.set_xlabel("$\\log_{10}$ cover (sparse $\\to$ diffuse)"); ax.set_ylabel("$\\log_{10}$ src/clump (clustering)")
    ax.set_title("Harvest ratio $\\log_2(U_{rec}/U_{base})$ (>0 = recruitment wins)")
    fig.colorbar(im, ax=ax, label="$\\log_2$ harvest ratio"); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_phasediagram.pdf"))

    # (2) marginals
    axes_cols = [("cover", True), ("src_per_clump", True), ("phi", True), ("alpha", False), ("z", False)]
    fig2, axx = plt.subplots(1, len(axes_cols), figsize=(3 * len(axes_cols), 3.2), sharey=True)
    for a, (col, lg) in zip(axx, axes_cols):
        c, r, mv = binned_ratio(df, col, log=lg)
        a.axhline(1, color="0.6", ls=":"); a.plot(c, r, "o-", color="#1f4e79", label="harvest")
        a.plot(c, mv, "s--", color="#c0392b", lw=1, label="move/harvest")
        if lg: a.set_xscale("log")
        a.set_xlabel(col); a.set_title(col)
    axx[0].set_ylabel("ratio (recruit/baseline)"); axx[-1].legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGDIR, "fig_q4_marginals.pdf"))

    # (3) importance = tercile swing of pooled harvest ratio
    print("\nper-axis importance (|log2| swing of pooled harvest ratio across terciles):")
    imp = {}
    for col in ["cover", "src_per_clump", "phi", "alpha", "z", "ars_ratio", "N", "deplete_ticks", "s_rel"]:
        if df[col].nunique() < 3:
            continue
        q = pd.qcut(df[col], 3, duplicates="drop")
        vals = df.groupby(q, observed=True).apply(pooled).values
        imp[col] = float(np.log2(max(vals) / max(min(vals), 1e-9)))
    for col, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {col:14s} {v:5.2f}")

    # (3b) MOVEMENT-COST drivers (distinct metric): absolute recruit steps/harvest, |log2| tercile swing
    def mvcost(g):
        return float(g.sumD_rec.sum() / max(g.sumU_rec.sum(), 1e-12))
    print("\nper-axis MOVEMENT-COST importance (|log2| swing of recruit steps/harvest):")
    impm = {}
    for col in ["alpha", "cover", "src_per_clump", "phi", "z", "ars_ratio", "N", "deplete_ticks"]:
        if df[col].nunique() < 3:
            continue
        q = pd.qcut(df[col], 3, duplicates="drop")
        vals = df.groupby(q, observed=True).apply(mvcost).values
        impm[col] = float(np.log2(max(vals) / max(min(vals), 1e-12)))
    for col, v in sorted(impm.items(), key=lambda kv: -kv[1]):
        print(f"  {col:14s} {v:5.2f}")
    qa = pd.qcut(df.alpha, min(5, df.alpha.nunique()), duplicates="drop")
    va = df.groupby(qa, observed=True).apply(mvcost)
    print("  recruit steps/harvest by alpha quantile: " + ", ".join(f"{v:.1f}" for v in va.values) +
          "  (orders-of-magnitude span => alpha drives movement cost)")

    # (4) bootstrap headline: sparsest-clustered quartile vs most-diffuse quartile
    def boot(mask, n=2000):
        sub = df[mask]; out = []
        for _ in range(n):
            s = sub.sample(len(sub), replace=True)
            out.append(pooled(s))
        return np.percentile(out, [2.5, 50, 97.5])
    sc = (df.cover < df.cover.quantile(.25)) & (df.src_per_clump > df.src_per_clump.quantile(.75))
    di = df.cover > df.cover.quantile(.75)
    lo, md, hi = boot(sc); print(f"\nsparse+clustered harvest ratio = {md:.2f} [95% CI {lo:.2f},{hi:.2f}]  (n={sc.sum()})")
    lo, md, hi = boot(di); print(f"diffuse harvest ratio          = {md:.2f} [95% CI {lo:.2f},{hi:.2f}]  (n={di.sum()})")
    print(f"\nwrote {FIGDIR}/fig_phasediagram.pdf, fig_q4_marginals.pdf")


if __name__ == "__main__":
    main()
