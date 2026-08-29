#!/usr/bin/env python3
"""
Levy-optimality on CLUSTERED vs SCATTERED fields (paper limit iii). The classical Levy-optimality
results are derived for sparse, RANDOMLY SCATTERED targets; our efficient regime is sparse AND
CLUSTERED. "Sparse" and "clustered" are not the same property, so the two need not join automatically.
Here we test, in our own model, whether heavy-tailed exploration is (near-)optimal for clustered
fields too: sweep the Levy exponent alpha and measure steady-state harvest, for a clustered field
(few clumps) vs a scattered field (targets spread ~uniformly).

Smaller alpha = heavier tail (longer jumps). If harvest peaks at heavy-tailed alpha for the clustered
field, Levy is (empirically, in this model) optimal for clustered targets too -- closing the gap
for our system (the general search-theory result remains open).

CPU only. Usage: python3 levy_field_test.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def harvest(alpha, clump_count, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=args.rho_r, phi=args.phi,
                              theta=args.theta, step=args.step, pull=args.pull,
                              consumption=args.consumption, capacity=args.capacity,
                              converge=True, seed=args.seed, device="cpu",
                              levy_alpha=alpha, exploit_step=args.exploit_step,
                              clump_count=clump_count, clump_radius=args.clump_radius)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - args.tail) * args.T)
    consumed = []
    for t in range(args.T):
        sim.step()
        if t >= t0:
            consumed.append(float(sim._consumed.mean()))     # resource removed this tick (throughput)
    return float(np.mean(consumed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=150); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--B", type=int, default=20); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--rho_r", type=float, default=3.0); ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=0.20)
    ap.add_argument("--step", type=float, default=20.0); ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0); ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--clump-radius", type=float, default=25.0, dest="clump_radius")
    ap.add_argument("--clustered-clumps", type=int, default=3, dest="clustered")
    ap.add_argument("--scattered-clumps", type=int, default=15, dest="scattered")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="levy_field")
    args = ap.parse_args()
    print(f"levy x field: N={args.N} theta={args.theta} clustered={args.clustered}clumps "
          f"scattered={args.scattered}clumps  alphas={args.alphas}", flush=True)
    res = {}
    for name, cc in [("clustered", args.clustered), ("scattered", args.scattered)]:
        h = np.array([harvest(a, cc, args) for a in args.alphas])
        res[name] = h
        astar = args.alphas[int(np.argmax(h))]
        print(f"  {name:9s} (clumps={cc:2d}): harvest by alpha = "
              f"{np.array2string(h, precision=3)}  -> alpha*={astar}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, alphas=np.array(args.alphas), clustered=res["clustered"], scattered=res["scattered"])
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
