#!/usr/bin/env python3
"""Q4 COST re-measurement on the bug-fixed engine (clean sumD; the Levy-overflow inf is gone).
Movement cost (steps-per-target) and communication (messages/target) vs the Levy index alpha, paired
recruit (theta_lo) vs no-recruit baseline (theta_base) at a representative sparse-clustered operating
point (the regime recruitment operates in; nuisances = the alpha=1 phase-grid FIX values). Establishes:
  (i) is alpha the dominant driver of movement cost?  (ii) does recruitment save/cost movement per target?
  (iii) is the message cost governed by connectivity, not alpha?  Reports from scratch."""
import os, sys, time, argparse
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
import numpy as np, torch
sys.path.insert(0, _P.CODE)
from mobile_cascade import MobileCascade, MobileCascadeConfig

DATADIR = os.path.join(_P.DATA, "q4_cost_v15")
FIX = dict(phi=0.0173, rho_r=3.5, p=0.625, N=120, K=23, ars_ratio=0.1, deplete_ticks=31.65, pull=0.1)
# representative sparse, clustered field (recruitment-favourable regime)
COVER, SPC = 0.02, 7.0


@torch.no_grad()
def ledger(theta, alpha, nclump, patch, args):
    cfg = MobileCascadeConfig(
        B=args.B, N=FIX["N"], K=FIX["K"], L=args.L, rho_r=FIX["rho_r"], p=FIX["p"], phi=FIX["phi"],
        theta=theta, theta_dist="const", theta_std=0.0,
        step=args.step, exploit_step=FIX["ars_ratio"] * args.step, pull=FIX["pull"], levy_alpha=alpha,
        clump_count=nclump, clump_radius=patch * args.L,
        capacity=FIX["deplete_ticks"] * args.consumption, consumption=args.consumption,
        converge=True, device=args.device, seed=args.seed)
    o = MobileCascade(cfg).run(args.T)
    t0 = int((1.0 - args.tail) * args.T); Tw = args.T - t0
    U = o["consumed"][:, t0:].sum(1).cpu().numpy() / (args.consumption * FIX["N"] * Tw)
    D = o["distance"][:, t0:].sum(1).cpu().numpy() / (FIX["N"] * Tw * args.step)   # steps (in step-units)
    Ma = o["msgs_activation"][:, t0:].sum(1).cpu().numpy() / (FIX["N"] * Tw)
    return U, D, Ma


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta_lo", type=float, default=0.20); ap.add_argument("--theta_base", type=float, default=0.99)
    ap.add_argument("--B", type=int, default=2048); ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--tail", type=float, default=0.5); ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--step", type=float, default=20.0); ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(DATADIR, exist_ok=True)
    nclump = max(1, int(round(FIX["K"] / SPC))); patch = float(np.sqrt(COVER / (nclump * np.pi)))
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); rows = []
    print(f"q4 cost: field cover={COVER} spc={SPC} (nclump={nclump}); alphas={args.alphas}", flush=True)
    print(f"{'alpha':>6} {'spt_rec':>10} {'spt_base':>10} {'mc_ratio':>9} {'msg/tgt':>9} {'U_rec':>7} {'U_base':>7} {'infD':>5}")
    for a in args.alphas:
        Ur, Dr, Mar = ledger(args.theta_lo, a, nclump, patch, args)
        Ub, Db, _ = ledger(args.theta_base, a, nclump, patch, args)
        spt_rec = Dr.sum() / max(Ur.sum(), 1e-9); spt_base = Db.sum() / max(Ub.sum(), 1e-9)
        mc = spt_rec / max(spt_base, 1e-9); msg = Mar.sum() / max(Ur.sum(), 1e-9)
        ninf = int(np.isinf(Dr).sum() + np.isinf(Db).sum())
        rows.append(dict(alpha=a, spt_rec=spt_rec, spt_base=spt_base, mc_ratio=mc, msg_per_tgt=msg,
                         U_rec=float(Ur.sum()), U_base=float(Ub.sum()), inf=ninf))
        print(f"{a:6.2f} {spt_rec:10.1f} {spt_base:10.1f} {mc:9.3f} {msg:9.1f} {Ur.sum():7.2f} {Ub.sum():7.2f} {ninf:5d}", flush=True)
    import json
    np.savez(os.path.join(DATADIR, "q4_cost_v15.npz"),
             alphas=np.array([r["alpha"] for r in rows]),
             spt_rec=np.array([r["spt_rec"] for r in rows]), spt_base=np.array([r["spt_base"] for r in rows]),
             mc_ratio=np.array([r["mc_ratio"] for r in rows]), msg_per_tgt=np.array([r["msg_per_tgt"] for r in rows]))
    vram = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0
    print(f"\n=== q4 cost done: {time.time()-t0:.0f}s, VRAM {vram:.2f}GB, all-finite={all(r['inf']==0 for r in rows)} ===")


if __name__ == "__main__":
    main()
