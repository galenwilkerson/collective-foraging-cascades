#!/usr/bin/env python3
"""
Local SOC test: is the criticality *local* (in recruited clusters) even though the global
average is subcritical? Measures the distribution of ACTIVE-CLUSTER sizes (connected components
of the active subgraph) over time -- power-law cluster sizes = scale-free local structure =
local SOC -- plus the largest-active-cluster fraction Psi and the number of clusters.

Runs a few theta at fixed (rho_r, phi) so we can see whether any operating point yields
power-law local clusters. Saves to sweeps/local_soc.npz for the notebook.

Usage: python3 local_soc_test.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def cluster_stats(adj, active):
    """Active-subgraph connected components. Returns (flattened cluster sizes, largest-frac per replica)."""
    B, N, _ = adj.shape
    dev = adj.device
    conn = (adj & active.unsqueeze(1) & active.unsqueeze(2))
    conn = conn | torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
    labels = torch.arange(N, device=dev).float().unsqueeze(0).expand(B, N).clone()
    INF = float(N + 1)
    for _ in range(N):
        lab_j = labels.unsqueeze(1).expand(B, N, N)
        new = torch.where(conn, lab_j, torch.full((), INF, device=dev)).min(dim=2).values
        if torch.equal(new, labels):
            break
        labels = new
    lab = labels.long()
    sizes, largest = [], np.zeros(B)
    for b in range(B):
        act = active[b]
        if int(act.sum()) == 0:
            continue
        c = torch.bincount(lab[b][act], minlength=N)
        c = c[c > 0]
        sizes.append(c.cpu().numpy())
        largest[b] = float(c.max()) / N
    return (np.concatenate(sizes) if sizes else np.array([])), largest


@torch.no_grad()
def run_setting(theta, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=args.rho_r, phi=args.phi,
                              theta=theta, step=args.step, pull=args.pull,
                              consumption=args.consumption, capacity=args.capacity,
                              converge=True, seed=args.seed,
                              levy_alpha=args.levy, exploit_step=args.exploit_step)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - args.tail) * args.T)
    all_sizes, psi = [], []
    for t in range(args.T):
        sim.step()
        if t % args.every == 0 or t >= t0:
            sz, lg = cluster_stats(sim._adj, sim._active_state)
            psi.append(lg.mean())
            if t >= t0:
                all_sizes.append(sz)
    sizes = np.concatenate(all_sizes) if all_sizes else np.array([])
    sizes = sizes[sizes >= 1]
    return sizes.astype(np.float32), np.array(psi, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=150)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=700)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--rho_r", type=float, default=3.0)
    ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.1, 0.2, 0.35])
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--levy", type=float, default=None, help="Levy exponent (exploration); None = random walk")
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="local_soc")
    args = ap.parse_args()

    print(f"local SOC: N={args.N} K={args.K} rho_r={args.rho_r} phi={args.phi} thetas={args.thetas}",
          flush=True)
    save = {"thetas": np.array(args.thetas), "N": np.array([args.N])}
    for th in args.thetas:
        sizes, psi = run_setting(th, args)
        save[f"sizes_{th:.2f}"] = sizes
        save[f"psi_{th:.2f}"] = psi
        big = (sizes >= 2)
        print(f"  theta={th:.2f}: {len(sizes)} clusters, mean size={sizes.mean():.1f}, "
              f"max={sizes.max():.0f}, frac>=2={big.mean():.2f}, Psi_ss={psi[len(psi)//2:].mean():.2f}",
              flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, **save)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
