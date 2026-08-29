#!/usr/bin/env python3
"""
Resource-distribution phase diagram: WHERE does recruitment pay?

Sweeps two axes of the resource layout and, at each cell, compares a recruiting swarm
(theta_lo) to a no-recruitment baseline (theta=0.99), same everything else:

    patchiness  = clump_radius / L   (dimensionless: tight clumps -> diffuse ~ uniform)  [x-axis]
    abundance   = phi = K (S/L)^2    (dimensionless coverage)                            [y-axis]

Reports, per cell, the recruit/baseline RATIO for:
    harvest       U_recruit / U_base          (throughput)
    eta_bit       (U/M_act)_recruit / ...      (activation-comms efficiency)

Ratio > 1 => coordination pays. Places 'sparse clumps' as one corner of a general map.
Uses the four raw ledgers; Levy exploration; clumped sources (clump_count fixed).

Usage: python3 resource_regimes.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def ledgers(theta, phi, patchiness, args):
    # patchiness is dimensionless (fraction of L); convert to a world-unit clump radius like r, S
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, phi=phi, theta=theta,
                              capacity=args.capacity, consumption=args.consumption,
                              levy_alpha=args.levy, clump_count=args.clumps, clump_radius=patchiness * args.L,
                              step=args.step, pull=args.pull, seed=args.seed)
    sim = MobileCascade(cfg)
    out = sim.run(args.T)
    t0 = int((1.0 - args.tail) * args.T); Tw = args.T - t0
    U = out["consumed"][:, t0:].sum().item() / (args.consumption * args.N * Tw)
    D = out["distance"][:, t0:].sum().item() / (args.N * Tw * args.step)
    Ma = out["msgs_activation"][:, t0:].sum().item() / (args.N * Tw)
    return U, D, Ma


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--rho_r", type=float, default=3.0)
    ap.add_argument("--theta-lo", type=float, default=0.20, dest="theta_lo")
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--clumps", type=int, default=3)
    ap.add_argument("--patchiness", type=float, nargs="+", default=[0.025, 0.075, 0.19, 0.5])  # clump_radius/L (tight->diffuse)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.005, 0.02, 0.08])    # abundance
    ap.add_argument("--capacity", type=float, default=200.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--levy", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="resource_regimes")
    args = ap.parse_args()

    patch = np.array(args.patchiness, dtype=float); phis = np.array(args.phis, dtype=float)
    harvest = np.zeros((len(phis), len(patch)))
    eta_bit = np.zeros((len(phis), len(patch)))
    print(f"resource regimes: {len(phis)} phi x {len(patch)} patchiness (clump_radius/L), clumps={args.clumps}, "
          f"K={args.K}, capacity={args.capacity}, levy={args.levy}", flush=True)
    for i, phi in enumerate(phis):
        for j, p in enumerate(patch):
            Ur, Dr, Mr = ledgers(args.theta_lo, phi, p, args)      # recruiting
            Ub, Db, Mb = ledgers(0.99, phi, p, args)               # no-recruitment baseline
            eps = 1e-9
            harvest[i, j] = Ur / (Ub + eps)
            # NOTE: harvest/distance (the old eta_move) was removed -- it is the degenerate
            # fixed-horizon ratio the project retired (rewards inactivity, divides by ~0 in diffuse
            # cells). Throughput (harvest) and comms-efficiency (eta_bit) are the kept metrics.
            eta_bit[i, j] = (Ur / (Mr + eps)) / (Ub / (Mb + eps) + eps)
            print(f"  phi={phi:.3f} patchiness={p:.3f}: harvest x{harvest[i,j]:.2f}  "
                  f"eta_bit x{eta_bit[i,j]:.2f}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, f"{args.tag}.npz"), patchiness=patch, phis=phis,
             harvest=harvest, eta_bit=eta_bit)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ext = [patch.min(), patch.max(), phis.min(), phis.max()]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, Z, ttl in [(axes[0], harvest, "harvest ratio"),
                           (axes[1], eta_bit, "comms-eff ratio")]:
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=0, vmax=2)
            ax.set_xlabel("patchiness  (clump radius / L, tight -> diffuse)"); ax.set_ylabel(r"abundance $\phi$")
            ax.set_title(f"{ttl}  (>1 = recruit pays)"); fig.colorbar(im, ax=ax)
        fig.suptitle("Where does recruitment pay? resource patchiness x abundance")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"{args.tag}.png"), dpi=120)
        print(f"wrote {args.tag}.png", flush=True)
    except ImportError:
        pass
    print(f"wrote {args.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
