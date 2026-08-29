#!/usr/bin/env python3
"""
SOC-cycle test for the minimal depleting mobile-cascade model.

Records, over time: activity, branching sigma, cascade (avalanche) size, connectivity, and
remaining resource -- to test whether the system travels the SOC.pdf cycle
(explore -> recruit -> consume -> explore) and whether the recruitment avalanches are
scale-free (power-law), the falsifiable SOC signature.

Runs the COUPLED model (recruitment -> convergence) and an ABLATED control (converge=False,
no topology-geometry coupling) for Q3, and saves everything to an .npz that the notebook plots.

Usage:
    python3 mobile_soc_test.py            # -> sweeps/mobile_soc.npz
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def run(converge, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=args.rho_r, phi=args.phi,
                              theta=args.theta, step=args.step, pull=args.pull,
                              consumption=args.consumption, capacity=args.capacity,
                              converge=converge, seed=args.seed,
                              levy_alpha=args.levy, exploit_step=args.exploit_step)
    sim = MobileCascade(cfg)
    out = sim.run(args.T)
    return {k: v.cpu().numpy() for k, v in out.items()}, sim.N


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--rho_r", type=float, default=3.0)
    ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=0.15)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--levy", type=float, default=None, help="Levy exponent (exploration); None = random walk")
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--n-show", type=int, default=4, help="individual replica trajectories to save")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="mobile_soc")
    args = ap.parse_args()

    print(f"SOC test: N={args.N} K={args.K} B={args.B} T={args.T} rho_r={args.rho_r} "
          f"theta={args.theta} phi={args.phi}", flush=True)
    save = {}
    for label, conv in [("coupled", True), ("ablated", False)]:
        out, N = run(conv, args)
        af = out["active_count"] / N                 # (B,T) active fraction
        for key, arr in [("active_frac", af), ("sigma", out["sigma"]),
                         ("cascade_size", out["cascade_size"]), ("z_global", out["z_global"]),
                         ("z_local", out["z_local"]), ("resource", out["resource"]),
                         ("exploring_fraction", out["exploring_fraction"])]:
            save[f"{label}_{key}"] = arr[: args.n_show]           # a few replicas (for the cycle)
            save[f"{label}_{key}_mean"] = arr.mean(axis=0)        # ensemble mean
        # avalanche sizes: all recruited-cascade sizes, all ticks x replicas (for the distribution)
        save[f"{label}_avalanches"] = out["cascade_size"].reshape(-1).astype(np.float32)
        print(f"  {label}: mean active_frac={af.mean():.3f}, mean cascade={out['cascade_size'].mean():.2f}, "
              f"max cascade={out['cascade_size'].max():.0f}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    meta = dict(N=args.N, K=args.K, B=args.B, T=args.T, rho_r=args.rho_r, phi=args.phi,
                theta=args.theta, step=args.step, pull=args.pull, consumption=args.consumption,
                capacity=args.capacity, n_show=args.n_show)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, t=np.arange(args.T), meta=np.array([str(meta)]), **save)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
