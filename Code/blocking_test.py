#!/usr/bin/env python3
"""
Is fraction-rule BLOCKING actually the self-limiter, or is it resource DEPLETION?

The paper states blocking is the intrinsic self-limitation: convergence raises degree, a/k falls
below theta, recruitment is blocked, agents are released to exploration. A first ablation at the
paper's operating point (z≈6.6) showed the opposite: with depletion OFF the swarm saturates at
~98-100% active and blocked→0. But z≈6.6 sits INSIDE the cascade window (below the blocking
boundary ~2/theta), so blocking is not expected to bite there.

This test sweeps connectivity z across and beyond the window, with depletion ON vs OFF:
  * depletion OFF isolates blocking as the ONLY possible self-limiter.
  * if active_frac stays ~1.0 at low z and FALLS at high z -> blocking is real, but only above
    the blocking boundary.
  * if active_frac stays ~1.0 at ALL z -> blocking never self-limits and the paper's mechanism
    sentence is mis-attributed.

Saves sweeps/blocking_test.npz.  Usage: python3 blocking_test.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def run(rho_r, theta, deplete, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=rho_r, phi=args.phi,
                              theta=theta, p=args.p, levy_alpha=args.alpha,
                              clump_count=args.clumps, clump_radius=args.clump_radius,
                              capacity=(args.cap if deplete else 1e9), consumption=args.cons,
                              converge=True, seed=args.seed)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - args.tail) * args.T)
    acc = {k: [] for k in ["active", "blocked", "isolated", "reverted", "casc", "harv", "move", "comm"]}
    for t in range(args.T):
        sim.step()
        if t >= t0:
            s = sim._snapshot()
            acc["active"].append(s["active_count"].mean().item())
            acc["blocked"].append(s["blocked"].mean().item())
            acc["isolated"].append(s["isolated"].mean().item())
            acc["reverted"].append(s["reverted"].mean().item())
            acc["casc"].append(s["cascade_size"].mean().item())
            acc["harv"].append(s["consumed"].mean().item())
            acc["move"].append(s["distance"].mean().item())
            acc["comm"].append((s["msgs_beacon"] + s["msgs_activation"]).mean().item())
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    return dict(active_frac=m["active"] / args.N, blocked_frac=m["blocked"] / args.N,
                isolated_frac=m["isolated"] / args.N, reverted_frac=m["reverted"] / args.N,
                casc_frac=m["casc"] / args.N, harvest=m["harv"],
                eff=m["harv"] / max(m["move"] + m["comm"], 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.1, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.18, 0.30])
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--phi", type=float, default=0.02)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clumps", type=int, default=6)
    ap.add_argument("--clump_radius", type=float, default=40.0)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--cons", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    args = ap.parse_args()
    keys = ["active_frac", "blocked_frac", "isolated_frac", "reverted_frac", "casc_frac", "harvest", "eff"]
    out = {}
    print(f"blocking test: rho_r={args.rhos} (z=pi*p*rho_r) x theta={args.thetas} x depletion ON/OFF",
          flush=True)
    for theta in args.thetas:
        for dep in (True, False):
            tag = f"th{theta:.2f}_{'dep' if dep else 'nodep'}"
            g = {k: np.zeros(len(args.rhos)) for k in keys}
            print(f"--- theta={theta}  depletion={'ON' if dep else 'OFF'}  (blocking boundary z~{2/theta:.1f}) ---",
                  flush=True)
            for i, r in enumerate(args.rhos):
                res = run(r, theta, dep, args)
                for k in keys:
                    g[k][i] = res[k]
                z = np.pi * args.p * r
                print(f"  rho_r={r:5.1f} z~{z:5.1f} | active={res['active_frac']:.2f} "
                      f"BLOCKED={res['blocked_frac']:.2f} isolated={res['isolated_frac']:.2f} "
                      f"reverted={res['reverted_frac']:.3f} | harvest={res['harvest']:.2f} eff={res['eff']:.4f}",
                      flush=True)
            for k in keys:
                out[f"{tag}_{k}"] = g[k]
    out["rhos"] = np.array(args.rhos); out["thetas"] = np.array(args.thetas)
    out["z"] = np.pi * args.p * np.array(args.rhos)
    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "blocking_test.npz")
    np.savez(p, **out)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
