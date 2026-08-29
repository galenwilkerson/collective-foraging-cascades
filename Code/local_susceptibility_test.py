#!/usr/bin/env python3
"""
Local-susceptibility test (paper limit ii): a second, independent criticality signature. Sweep the
connectivity control rho_r (-> mean degree z); at each, measure the active-CLUSTER-size susceptibility
    chi = <S^2> / <S>   (size-weighted mean cluster size, EXCLUDING the largest cluster per replica),
the standard percolation susceptibility. A critical system has a chi peak; we test whether it peaks
INSIDE the cascade window, complementing sigma_local (branching) and the size exponent tau.

CPU only. Usage: python3 local_susceptibility_test.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402
from local_soc_test import cluster_stats  # reuse connected-component sizes  # noqa: E402


@torch.no_grad()
def component_sizes(adj, active):
    """List of per-replica arrays of active-cluster sizes (reuse cluster_stats returns flat; here we
    need per-replica so we recompute labels)."""
    B, N, _ = adj.shape; dev = adj.device
    conn = (adj & active.unsqueeze(1) & active.unsqueeze(2)) | torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
    labels = torch.arange(N, device=dev).float().unsqueeze(0).expand(B, N).clone()
    INF = float(N + 1)
    for _ in range(N):
        new = torch.where(conn, labels.unsqueeze(1).expand(B, N, N), torch.full((), INF, device=dev)).min(dim=2).values
        if torch.equal(new, labels):
            break
        labels = new
    lab = labels.long(); out = []
    for b in range(B):
        act = active[b]
        if int(act.sum()) == 0:
            out.append(np.array([])); continue
        c = torch.bincount(lab[b][act], minlength=N); c = c[c > 0]
        out.append(c.cpu().numpy())
    return out


@torch.no_grad()
def run_rho(rho_r, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=rho_r, phi=args.phi,
                              theta=args.theta, step=args.step, pull=args.pull,
                              consumption=args.consumption, capacity=args.capacity,
                              converge=True, seed=args.seed, device="cpu",
                              levy_alpha=args.levy, exploit_step=args.exploit_step)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - args.tail) * args.T)
    chis, zg, casc = [], [], []
    for t in range(args.T):
        sim.step()
        if t < t0:
            continue
        zg.append(float(sim._z_global.mean()))
        casc.append(float(sim._cascade_size.mean()) / args.N)
        for sizes in component_sizes(sim._adj, sim._active_state):
            if len(sizes) == 0:
                continue                                  # no active agents this replica-tick
            s = np.sort(sizes)[:-1].astype(float)         # exclude the largest (spanning) cluster
            if s.sum() <= 0:
                chis.append(0.0)                          # only one (spanning) cluster -> no susceptibility mass
            else:
                chis.append((s ** 2).sum() / s.sum())
    return (float(np.mean(chis)) if chis else np.nan,
            float(np.mean(zg)), float(np.mean(casc)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=150); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--B", type=int, default=20); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--rho_grid", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    ap.add_argument("--phi", type=float, default=0.05); ap.add_argument("--theta", type=float, default=0.20)
    ap.add_argument("--step", type=float, default=20.0); ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0); ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--levy", type=float, default=None)
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="local_susceptibility")
    args = ap.parse_args()
    mode = "Levy(a=%.2f)" % args.levy if args.levy is not None else "random walk"
    print(f"local susceptibility: N={args.N} theta={args.theta} exploration={mode}", flush=True)
    chi, zg, casc = [], [], []
    for rho in args.rho_grid:
        c, z, cf = run_rho(rho, args)
        chi.append(c); zg.append(z); casc.append(cf)
        print(f"  rho_r={rho:5.1f}  z_global={z:5.2f}  cascade_frac={cf:.3f}  chi_local={c:.2f}", flush=True)
    chi = np.array(chi); zg = np.array(zg)
    pk = int(np.nanargmax(chi))
    print(f"  --> chi peaks at rho_r={args.rho_grid[pk]:.1f} (z_global={zg[pk]:.2f}, "
          f"cascade_frac={casc[pk]:.3f})", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, rho_grid=np.array(args.rho_grid), chi=chi, z_global=zg, cascade_frac=np.array(casc),
             theta=np.array([args.theta]))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
