#!/usr/bin/env python3
"""
Re-test threshold heterogeneity under the COST-PER-JOB metric (steps/target, bits/target).

The earlier conclusion -- "threshold heterogeneity lowers efficiency" -- rested on
eff = harvest/(movement+communication) at a FIXED HORIZON. That metric is now known to be flawed:
it sums incommensurate units (and is ~90% movement-dominated, so communication barely registered),
and being fixed-horizon it rewards inactivity. So the conclusion must be re-derived with the
ungameable measure: cost to consume a FIXED number of targets.

Sweeps (theta mean x spread) for gaussian and uniform threshold laws, reporting per cell:
  steps/target                (movement cost of the job; LOWER = better)
  beacon/target, activation/target, bits/target   (communication, split by ping type)
  ticks/target                (time cost -- why beacons scale with job duration)
All evaluated at a COMMON job size X so every cell is compared on the same amount of work.

Saves sweeps/threshold_cost_{gaussian,uniform}.npz.   Usage: python3 threshold_cost.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def trajectory(dist, theta, spread, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, phi=args.phi,
                              theta=theta, theta_dist=dist, theta_std=spread,
                              capacity=args.capacity, consumption=args.consumption,
                              levy_alpha=args.alpha, clump_count=args.clumps,
                              clump_radius=args.clump_radius, step=args.step, pull=args.pull,
                              seed=args.seed)
    out = MobileCascade(cfg).run(args.T)
    return dict(tgt=(out["consumed"].mean(0).cumsum(0) / args.capacity).cpu().numpy(),
                stp=(out["distance"].mean(0).cumsum(0) / args.step).cpu().numpy(),
                bec=out["msgs_beacon"].mean(0).cumsum(0).cpu().numpy(),
                act=out["msgs_activation"].mean(0).cumsum(0).cpu().numpy(),
                exp=(out["dist_explore"].mean(0).cumsum(0) / args.step).cpu().numpy(),
                cnv=(out["dist_converge"].mean(0).cumsum(0) / args.step).cpu().numpy())


def at_job(tr, X):
    i = np.searchsorted(tr["tgt"], X)
    if i >= len(tr["tgt"]):
        return None
    return {k: float(v[i]) for k, v in tr.items() if k != "tgt"} | {"ticks": float(i + 1)}


def sweep(dist, thetas, spreads, args):
    trs = {(i, j): trajectory(dist, th, sp, args)
           for i, th in enumerate(thetas) for j, sp in enumerate(spreads)}
    X = args.job_frac * min(t["tgt"][-1] for t in trs.values())     # common job all cells complete
    keys = ["steps", "beacon", "activation", "bits", "ticks", "explore", "converge"]
    G = {k: np.full((len(thetas), len(spreads)), np.nan) for k in keys}
    print(f"=== {dist}: cost to consume X={X:.1f} targets (LOWER = better) ===", flush=True)
    for (i, j), tr in trs.items():
        c = at_job(tr, X)
        if c is None:
            continue
        G["steps"][i, j] = c["stp"] / X
        G["beacon"][i, j] = c["bec"] / X
        G["activation"][i, j] = c["act"] / X
        G["bits"][i, j] = (c["bec"] * args.bits_beacon + c["act"] * args.bits_activation) / X
        G["ticks"][i, j] = c["ticks"] / X
        G["explore"][i, j] = c["exp"] / X
        G["converge"][i, j] = c["cnv"] / X
        print(f"  theta={thetas[i]:.2f} spread={spreads[j]:.2f} | steps/tgt={G['steps'][i,j]:7.1f} "
              f"(explore {G['explore'][i,j]:6.1f} + travel {G['converge'][i,j]:6.1f}) | "
              f"bits/tgt={G['bits'][i,j]:7.1f} (beacon {G['beacon'][i,j]:6.1f} + act "
              f"{G['activation'][i,j]:6.1f}) | ticks/tgt={G['ticks'][i,j]:6.1f}", flush=True)
    # does spread help or hurt, per theta, on each axis independently?
    print("  --- effect of spread vs the homogeneous (spread=0) cell, same theta ---", flush=True)
    for i, th in enumerate(thetas):
        s0, b0 = G["steps"][i, 0], G["bits"][i, 0]
        best_s = np.nanargmin(G["steps"][i]); best_b = np.nanargmin(G["bits"][i])
        print(f"   theta={th:.2f}: steps best at spread={spreads[best_s]:.2f} "
              f"({G['steps'][i,best_s]:.1f} vs {s0:.1f} homogeneous, "
              f"{100*(G['steps'][i,best_s]/s0-1):+.1f}%) | "
              f"bits best at spread={spreads[best_b]:.2f} "
              f"({G['bits'][i,best_b]:.1f} vs {b0:.1f}, {100*(G['bits'][i,best_b]/b0-1):+.1f}%)",
              flush=True)
    return dict(thetas=np.array(thetas), spreads=np.array(spreads), job=X, **G)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--rho_r", type=float, default=2.1)
    ap.add_argument("--phi", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clumps", type=int, default=6)
    ap.add_argument("--clump_radius", type=float, default=40.0)
    ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--bits_beacon", type=float, default=1.0)
    ap.add_argument("--bits_activation", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--job_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gauss_means", type=float, nargs="+", default=[0.10, 0.18, 0.26, 0.30])
    ap.add_argument("--gauss_stds", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.08, 0.12])
    ap.add_argument("--unif_centers", type=float, nargs="+", default=[0.10, 0.18, 0.26])
    ap.add_argument("--unif_widths", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    g = sweep("gaussian", args.gauss_means, args.gauss_stds, args)
    np.savez(os.path.join(args.outdir, "threshold_cost_gaussian.npz"), **g)
    u = sweep("uniform", args.unif_centers, args.unif_widths, args)
    np.savez(os.path.join(args.outdir, "threshold_cost_uniform.npz"), **u)
    print("wrote sweeps/threshold_cost_{gaussian,uniform}.npz", flush=True)


if __name__ == "__main__":
    main()
