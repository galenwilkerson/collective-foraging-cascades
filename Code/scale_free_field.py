#!/usr/bin/env python3
"""
Empirical-contact test: drive the model with a SCALE-FREE (power-law) resource field, as real
resource fields are (e.g. Kalahari vegetation, beta<2), instead of a few Gaussian blobs.

For each imposed clump-size exponent (clump_beta) we:
  (1) measure the ACTUAL resource cluster-size exponent beta_res (from the realized clump sizes),
  (2) check whether recruitment still PAYS -- harvest ratio recruit/baseline (phase boundary survives?),
  (3) measure the swarm's EMERGENT active-cluster size exponent tau_active, and ask whether it TRACKS
      the imposed resource exponent.

Saves sweeps/scale_free_field.npz.
Usage: python3 scale_free_field.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402
from local_soc_test import cluster_stats                        # active-subgraph connected components


def ccdf_exponent(sizes, smin=2):
    """Fit P(S>=s) ~ s^-(exp-1); returns (exponent, R^2) with exponent = 1 + |CCDF slope|."""
    sizes = np.asarray(sizes, float); sizes = sizes[sizes >= smin]
    if sizes.size < 8:
        return np.nan, np.nan
    x = np.sort(sizes); c = 1.0 - np.arange(len(x)) / len(x)
    m = (x >= smin) & (x <= x.max() * 0.6) & (c > 0)
    if m.sum() < 6:
        return np.nan, np.nan
    b = np.polyfit(np.log10(x[m]), np.log10(c[m]), 1)
    pred = np.polyval(b, np.log10(x[m]))
    r2 = 1 - np.sum((np.log10(c[m]) - pred) ** 2) / np.sum((np.log10(c[m]) - np.log10(c[m]).mean()) ** 2)
    return 1.0 - b[0], r2                     # beta = 1 + |slope|  (slope is negative)


@torch.no_grad()
def resource_beta(sim):
    """Realized resource cluster-size exponent from the clump occupancies."""
    sizes = []
    for b in range(sim.B):
        cnt = torch.bincount(sim._clump_idx[b], minlength=sim._n_clumps).cpu().numpy()
        sizes.append(cnt[cnt > 0])
    return ccdf_exponent(np.concatenate(sizes))


@torch.no_grad()
def run(theta, input_beta, args, collect=False):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=args.rho_r, phi=args.phi, theta=theta,
                              capacity=args.capacity, consumption=args.consumption, levy_alpha=1.0,
                              clump_beta=input_beta, clump_radius=args.clump_radius,
                              step=args.step, pull=args.pull, seed=args.seed)
    sim = MobileCascade(cfg)
    t0 = int((1 - args.tail) * args.T); consumed = 0.0; sizes = []
    for t in range(args.T):
        sim.step()
        if t >= t0:
            consumed += sim._consumed.sum().item()
            if collect and t % 3 == 0:
                sz, _ = cluster_stats(sim._adj, sim._active_state)
                sizes.append(sz)
    U = consumed / (args.consumption * args.N * (args.T - t0) * sim.B)
    return sim, U, (np.concatenate(sizes) if sizes else np.array([]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--rho_r", type=float, default=3.0)
    ap.add_argument("--phi", type=float, default=0.01)
    ap.add_argument("--capacity", type=float, default=200.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--clump_radius", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="scale_free_field")
    args = ap.parse_args()

    print(f"scale-free field: input betas={args.betas}, K={args.K}, capacity={args.capacity}, Levy", flush=True)
    rows, save = [], {"input_beta": np.array(args.betas)}
    for ib in args.betas:
        sim_r, U_r, act = run(0.20, ib, args, collect=True)   # recruiting
        beta_res, r2b = resource_beta(sim_r)
        tau_active, r2t = ccdf_exponent(act)
        _, U_b, _ = run(0.99, ib, args)                       # no-recruitment baseline
        ratio = U_r / (U_b + 1e-9)
        rows.append((ib, beta_res, ratio, tau_active))
        save[f"act_sizes_{ib:.2f}"] = act.astype(np.float32)
        print(f"  input_beta={ib:.2f}: beta_res={beta_res:.2f} (R2={r2b:.2f}) | "
              f"harvest recruit/base={ratio:.2f} | tau_active={tau_active:.2f} (R2={r2t:.2f})", flush=True)
    save["summary"] = np.array(rows)   # cols: input_beta, beta_res, harvest_ratio, tau_active
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, f"{args.tag}.npz"), **save)
    print(f"wrote {args.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
