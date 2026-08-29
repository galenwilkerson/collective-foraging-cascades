#!/usr/bin/env python3
"""
Threshold-distribution sweep: does per-agent LTM-threshold HETEROGENEITY (vs a single uniform theta)
introduce useful non-uniformity into the cascade-swarm behaviour?

Motivation: with a homogeneous threshold, cascades are all-or-nothing (everyone recruits, then the
whole cluster reverts) -- no graded blocking / explore-exploit balance. A threshold distribution
(Watts 2002 uses one) may yield graded recruitment, a sustained exploring fraction, broader
scale-free cluster sizes, and different efficiency.

Two sweeps at a fixed cascade-window, sparse-clustered operating point:
  gaussian : theta_mean x theta_std   (per-agent theta ~ N(mean, std))
  uniform  : theta_center x half_width (per-agent theta ~ U[center-w, center+w])

Per cell, steady-state (tail) means of: harvest (consumed), movement, communication, efficiency
= harvest/(movement+comm), active & cascade fractions, exploring fraction (the balance), and the
LOCAL-criticality active-cluster-size distribution exponent (CCDF log-log slope) + R^2.

Saves sweeps/threshold_gaussian.npz and threshold_uniform.npz.
Usage: python3 threshold_sweep.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402
from local_soc_test import cluster_stats                        # noqa: E402


def powerlaw_ccdf(sizes):
    """CCDF log-log slope and R^2 for active-cluster sizes (>=2). Returns (slope, r2, mean, mx)."""
    s = sizes[sizes >= 2]
    if s.size < 20:
        return np.nan, np.nan, (sizes.mean() if sizes.size else 0.0), (sizes.max() if sizes.size else 0.0)
    x = np.sort(s)
    c = 1.0 - np.arange(len(x)) / len(x)
    m = (x <= x.max() * 0.5) & (c > 0)                          # avoid finite-size tail
    if m.sum() < 8:
        return np.nan, np.nan, sizes.mean(), sizes.max()
    b = np.polyfit(np.log10(x[m]), np.log10(c[m]), 1)
    pred = np.polyval(b, np.log10(x[m]))
    r2 = 1 - np.sum((np.log10(c[m]) - pred) ** 2) / np.sum((np.log10(c[m]) - np.log10(c[m]).mean()) ** 2)
    return float(b[0]), float(r2), float(sizes.mean()), float(sizes.max())


def run_cell(dist, theta, spread, args, deplete=True):
    # deplete=False -> capacity ~ infinite: sources never exhaust, isolating whether the
    # self-limitation comes from fraction-rule BLOCKING or merely from resource DEPLETION.
    cap = args.cap if deplete else 1e9
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=args.rho_r, phi=args.phi,
                              theta=theta, theta_dist=dist, theta_std=spread,
                              p=args.p, levy_alpha=args.alpha, clump_count=args.clumps,
                              clump_radius=args.clump_radius, capacity=cap,
                              consumption=args.cons, converge=True, seed=args.seed)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - args.tail) * args.T)
    harv, move, comm, act, casc, blk, iso, rev = [], [], [], [], [], [], [], []
    sizes_all = []
    for t in range(args.T):
        sim.step()
        if t >= t0:
            s = sim._snapshot()
            harv.append(s["consumed"].mean().item())
            move.append(s["distance"].mean().item())
            comm.append((s["msgs_beacon"] + s["msgs_activation"]).mean().item())
            act.append(s["active_count"].mean().item())
            casc.append(s["cascade_size"].mean().item())
            blk.append(s["blocked"].mean().item())
            iso.append(s["isolated"].mean().item())
            rev.append(s["reverted"].mean().item())
            sz, _ = cluster_stats(sim._adj, sim._active_state)
            if sz.size:
                sizes_all.append(sz)
    H, Mv, Cm = np.mean(harv), np.mean(move), np.mean(comm)
    A, Ca = np.mean(act), np.mean(casc)
    sizes = np.concatenate(sizes_all) if sizes_all else np.array([])
    slope, r2, mcl, mxcl = powerlaw_ccdf(sizes)
    return dict(harvest=H, move=Mv, comm=Cm, eff=H / max(Mv + Cm, 1e-9),
                active_frac=A / args.N, casc_frac=Ca / args.N, explore_frac=1 - A / args.N,
                blocked_frac=np.mean(blk) / args.N, isolated_frac=np.mean(iso) / args.N,
                reverted_frac=np.mean(rev) / args.N,
                tau=slope, r2=r2, mean_cluster=mcl, max_cluster=mxcl)


def sweep(dist, thetas, spreads, args, device, deplete=True):
    keys = ["harvest", "move", "comm", "eff", "active_frac", "casc_frac", "explore_frac",
            "blocked_frac", "isolated_frac", "reverted_frac",
            "tau", "r2", "mean_cluster", "max_cluster"]
    grids = {k: np.zeros((len(thetas), len(spreads))) for k in keys}
    tag = "depletion ON" if deplete else "depletion OFF (ablation)"
    print(f"=== {dist} sweep [{tag}]: theta={thetas} x spread={spreads} ===", flush=True)
    for i, th in enumerate(thetas):
        for j, sp in enumerate(spreads):
            r = run_cell(dist, th, sp, args, deplete=deplete)
            for k in keys:
                grids[k][i, j] = r[k]
            print(f"  theta={th:.2f} spread={sp:.2f} | active={r['active_frac']:.2f} "
                  f"BLOCKED={r['blocked_frac']:.2f} isolated={r['isolated_frac']:.2f} "
                  f"reverted={r['reverted_frac']:.3f} | eff={r['eff']:.3f} harvest={r['harvest']:.2f}",
                  flush=True)
    return dict(thetas=np.array(thetas), spreads=np.array(spreads), **grids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--rho_r", type=float, default=2.1, help="z = pi*p*rho_r ≈ 6.6 (cascade window)")
    ap.add_argument("--phi", type=float, default=0.02)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clumps", type=int, default=6, help="clustered resources (recruitment-informative)")
    ap.add_argument("--clump_radius", type=float, default=40.0)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--cons", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=320)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", action="store_true", default=True,
                    help="also run the depletion-OFF ablation (isolates blocking vs depletion)")
    ap.add_argument("--no-ablate", dest="ablate", action="store_false")
    ap.add_argument("--gauss_means", type=float, nargs="+", default=[0.10, 0.14, 0.18, 0.22, 0.26, 0.30])
    ap.add_argument("--gauss_stds", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.08, 0.12])
    ap.add_argument("--unif_centers", type=float, nargs="+", default=[0.10, 0.18, 0.26])
    ap.add_argument("--unif_widths", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"threshold sweep: N={args.N} rho_r={args.rho_r} phi={args.phi} clumps={args.clumps} "
          f"B={args.B} T={args.T} dev={device}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    modes = [(True, ""), (False, "_nodeplete")] if args.ablate else [(True, "")]
    for dep, suf in modes:
        g = sweep("gaussian", args.gauss_means, args.gauss_stds, args, device, deplete=dep)
        np.savez(os.path.join(args.outdir, f"threshold_gaussian{suf}.npz"), **g)
        print(f"wrote {args.outdir}/threshold_gaussian{suf}.npz", flush=True)
        u = sweep("uniform", args.unif_centers, args.unif_widths, args, device, deplete=dep)
        np.savez(os.path.join(args.outdir, f"threshold_uniform{suf}.npz"), **u)
        print(f"wrote {args.outdir}/threshold_uniform{suf}.npz", flush=True)


if __name__ == "__main__":
    main()
