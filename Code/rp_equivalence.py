#!/usr/bin/env python3
"""
Does only the PRODUCT z = pi * p * rho_r matter? (tests the paper's dimensional reduction, Sec 2.1)

The paper reduces connectivity to a single dimensionless number z = pi*p*rho_r = pi*p*N(r/L)^2.
But r and p are not obviously interchangeable:
  * lowering r  shrinks the neighbourhood -> graph stays geometrically LOCAL and clustered;
  * lowering p  keeps the same spatial reach but RANDOMLY THINS edges -> same extent, less clustering.
Both reduce z. If cascade/blocking/efficiency depend on r and p separately, the reduction needs
qualifying; if they collapse onto z alone, the reduction is validated.

For each target z we hold z fixed and reach it three ways: p = 1.0, 0.5, 0.25 with
rho_r = z / (pi * p). We then compare the observables that matter.

Saves sweeps/rp_equivalence.npz.  Usage: python3 rp_equivalence.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402
from local_soc_test import cluster_stats                        # noqa: E402


def run(z_target, p, args):
    rho_r = z_target / (np.pi * p)
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=rho_r, phi=args.phi,
                              theta=args.theta, p=p, levy_alpha=args.alpha,
                              clump_count=args.clumps, clump_radius=args.clump_radius,
                              capacity=args.cap, consumption=args.cons, converge=True,
                              step=args.step, pull=args.pull, seed=args.seed)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - args.tail) * args.T)
    acc = {k: [] for k in ["active", "casc", "blocked", "isolated", "reverted",
                           "harv", "move", "comm", "zmeas"]}
    sizes = []
    for t in range(args.T):
        sim.step()
        if t >= t0:
            s = sim._snapshot()
            acc["active"].append(s["active_count"].mean().item())
            acc["casc"].append(s["cascade_size"].mean().item())
            acc["blocked"].append(s["blocked"].mean().item())
            acc["isolated"].append(s["isolated"].mean().item())
            acc["reverted"].append(s["reverted"].mean().item())
            acc["harv"].append(s["consumed"].mean().item())
            acc["move"].append(s["distance"].mean().item())
            acc["comm"].append((s["msgs_beacon"] + s["msgs_activation"]).mean().item())
            acc["zmeas"].append(s["z_global"].mean().item())
            sz, _ = cluster_stats(sim._adj, sim._active_state)
            if sz.size:
                sizes.append(sz)
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    # clustering of the active-cluster sizes (local criticality signature)
    allsz = np.concatenate(sizes) if sizes else np.array([])
    mean_cl = float(allsz.mean()) if allsz.size else 0.0
    return dict(rho_r=rho_r, z_meas=m["zmeas"],
                active_frac=m["active"] / args.N, casc_frac=m["casc"] / args.N,
                blocked_frac=m["blocked"] / args.N, isolated_frac=m["isolated"] / args.N,
                reverted_frac=m["reverted"] / args.N, harvest=m["harv"],
                eff=m["harv"] / max(m["move"] + m["comm"], 1e-9), mean_cluster=mean_cl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zs", type=float, nargs="+", default=[4.7, 9.4, 25.0])
    ap.add_argument("--ps", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--phi", type=float, default=0.02)
    ap.add_argument("--theta", type=float, default=0.18)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clumps", type=int, default=6)
    ap.add_argument("--clump_radius", type=float, default=40.0)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--cons", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    args = ap.parse_args()
    keys = ["rho_r", "z_meas", "active_frac", "casc_frac", "blocked_frac", "isolated_frac",
            "reverted_frac", "harvest", "eff", "mean_cluster"]
    out = {k: np.zeros((len(args.zs), len(args.ps))) for k in keys}
    print("r-vs-p equivalence: is z = pi*p*rho_r the only thing that matters?", flush=True)
    for i, z in enumerate(args.zs):
        print(f"--- target z = {z} (blocking boundary 2/theta = {2/args.theta:.1f}) ---", flush=True)
        for j, p in enumerate(args.ps):
            r = run(z, p, args)
            for k in keys:
                out[k][i, j] = r[k]
            print(f"  p={p:.2f} rho_r={r['rho_r']:5.2f} | z_meas={r['z_meas']:5.2f} "
                  f"active={r['active_frac']:.3f} casc={r['casc_frac']:.3f} "
                  f"BLOCKED={r['blocked_frac']:.3f} | harvest={r['harvest']:.2f} eff={r['eff']:.4f} "
                  f"meanClust={r['mean_cluster']:.2f}", flush=True)
        # spread across p at fixed z = how badly the reduction fails
        for k in ["active_frac", "casc_frac", "blocked_frac", "eff"]:
            v = out[k][i]
            rng = (v.max() - v.min()) / max(abs(v.mean()), 1e-9)
            print(f"    spread across p in {k}: {100*rng:5.1f}% of mean", flush=True)
    out["zs"] = np.array(args.zs); out["ps"] = np.array(args.ps)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "rp_equivalence.npz")
    np.savez(path, **out)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
