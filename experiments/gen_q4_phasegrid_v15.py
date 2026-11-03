#!/usr/bin/env python3
"""Q4 phase diagram as a CONTROLLED alpha=1 grid (cover x src_per_clump), corrected engine -> v15.

Why: the Sobol run marginalizes over alpha, but harvest depends strongly on alpha (recruitment helps most
when solitary search is poor, i.e. Brownian alpha->2, least at the matched inverse-square alpha=1). The
phase-diagram caption states "computed at alpha=1", so the figure must be a controlled alpha=1 sweep, not an
alpha-marginal. Here alpha is fixed at 1.0 (mu=2, inverse-square Levy) and every other axis is held at its
Sobol median NEAR alpha=1 (homogeneous baseline s_rel=0); only cover (field sparsity) and src_per_clump
(clustering) vary on a grid. Paired recruit (theta_lo) vs no-recruit baseline (theta_base), same field+seed.
Raw per-replica harvest saved for downstream bootstrap CIs. Data on /mnt/newssd (standing rule)."""
import os, sys, time, argparse
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
import numpy as np, pandas as pd, torch
sys.path.insert(0, _P.CODE)
from mobile_cascade import MobileCascade, MobileCascadeConfig

DATADIR = os.path.join(_P.DATA, "q4_phasegrid_v15")
# nuisance axes held at Sobol medians near alpha=1 (homogeneous baseline: s_rel=0)
FIX = dict(phi=0.0173, rho_r=3.5, p=0.625, N=120, K=23, ars_ratio=0.1, deplete_ticks=31.65, pull=0.1)
COVER = np.geomspace(0.005, 0.8, 6)      # field sparsity axis (fraction of arena covered)
SPC = np.geomspace(1.0, 25.0, 6)         # clustering axis (sources per clump)


@torch.no_grad()
def ledger(theta, nclump, patch, args, baseline):
    cfg = MobileCascadeConfig(
        B=args.B, N=FIX["N"], K=FIX["K"], L=args.L, rho_r=FIX["rho_r"], p=FIX["p"], phi=FIX["phi"],
        theta=theta, theta_dist="const", theta_std=0.0,
        step=args.step, exploit_step=FIX["ars_ratio"] * args.step, pull=FIX["pull"], levy_alpha=1.0,
        clump_count=nclump, clump_radius=patch * args.L,
        capacity=FIX["deplete_ticks"] * args.consumption, consumption=args.consumption,
        converge=True, device=args.device, seed=args.seed)
    o = MobileCascade(cfg).run(args.T)
    t0 = int((1.0 - args.tail) * args.T); Tw = args.T - t0; den = args.consumption * FIX["N"] * Tw
    U = o["consumed"][:, t0:].sum(1).cpu().numpy() / den
    cs = float(o["cascade_size"].mean().item())
    return U, cs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta_lo", type=float, default=0.20); ap.add_argument("--theta_base", type=float, default=0.99)
    ap.add_argument("--B", type=int, default=4096); ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--tail", type=float, default=0.5); ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--step", type=float, default=20.0); ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="q4_phasegrid_v15")
    args = ap.parse_args()
    os.makedirs(DATADIR, exist_ok=True)
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); rows, raw = [], {}
    for ci, cover in enumerate(COVER):
        for si, spc in enumerate(SPC):
            nclump = max(1, int(round(FIX["K"] / spc)))
            patch = float(np.sqrt(cover / (nclump * np.pi)))
            if patch > 0.7:
                continue
            Ur, csr = ledger(args.theta_lo, nclump, patch, args, baseline=False)
            Ub, csb = ledger(args.theta_base, nclump, patch, args, baseline=True)
            key = f"c{ci}s{si}"
            raw[f"{key}__Urec"] = Ur.astype(np.float32); raw[f"{key}__Ubase"] = Ub.astype(np.float32)
            rows.append(dict(ci=ci, si=si, cover=float(cover), src_per_clump=float(spc),
                actual_cover=float(nclump * np.pi * patch ** 2), nclump=nclump, patch=patch,
                sumU_rec=float(Ur.sum()), sumU_base=float(Ub.sum()),
                ratio=float(Ur.sum() / max(Ub.sum(), 1e-12)), cascade_rec=csr, cascade_base=csb))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(DATADIR, f"{args.tag}.csv"), index=False)
    np.savez(os.path.join(DATADIR, f"{args.tag}_raw.npz"), **raw)
    vram = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0
    print(f"\n=== alpha=1 phase grid: {time.time()-t0:.0f}s, VRAM {vram:.2f}GB, {len(df)} cells, B={args.B} ===")
    print(f"baseline cascade mean={df.cascade_base.mean():.3f} (recruit {df.cascade_rec.mean():.1f})")
    hot = df.loc[df.ratio.idxmax()]
    print(f"hottest cell: cover={hot.cover:.3g} src/clump={hot.src_per_clump:.3g} ratio={hot.ratio:.2f}")
    print(f"min cell ratio={df.ratio.min():.2f}; cells with ratio>1: {(df.ratio>1).sum()}/{len(df)}")
    for lab, col in [("cover(sparse->diffuse)", "cover"), ("src_per_clump(clustering)", "src_per_clump")]:
        g = df.groupby(pd.qcut(df[col], 3, duplicates="drop"), observed=True).apply(
            lambda s: s.sumU_rec.sum() / max(s.sumU_base.sum(), 1e-12))
        print(f"  harvest ratio by {lab} tercile: " + ", ".join(f"{v:.2f}" for v in g.values))


if __name__ == "__main__":
    main()
