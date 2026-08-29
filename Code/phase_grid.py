#!/usr/bin/env python3
"""
Coarse-grained phase-diagram grid sweep over ALL kept dimensionless controls:

    rho_r  (connectivity, via mean degree z = pi*p*rho_r)   -- looped
    theta  (cascade threshold)                              -- folded into batch
    phi    (resource coverage / seeding)                    -- folded into batch

with R replicas per cell. Exploration is a near-Brownian random walk (alpha fixed; Levy is
out of scope). Steady-state (replenishing) dynamic episodes; active-weighted steady-state
observables answering Q1/Q2:

    sigma        cascade branching ratio  (sigma=1 = marginal; the cascade-window / SOC target)
    z_local      mean RGG degree of active agents  (self-tune to z_c ? or track z_global ?)
    z_global     realized mean degree over all agents
    active_frac  time-avg active fraction  (where recruitment actually fires)
    phi_expl     exploring fraction

Outputs, per phi, a (rho_r x theta) heatmap of sigma with the sigma=1 contour; full 3-D grids
saved to .npz for downstream analysis.

Usage:
    python3 phase_grid.py                 # coarse grid over rho_r x theta x phi
    python3 phase_grid.py --quick
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_swarm import BatchedSwarm, config_from_dimensionless, select_device  # noqa: E402

Z_C = 4.51


@torch.no_grad()
def run_rho(rho_r, thetas_batch, phis_batch, *, alpha, N, M, T, p, tail, device, seed):
    """One rho_r; (theta,phi) folded into the batch. Returns per-run (B,) steady-state obs."""
    B = thetas_batch.shape[0]
    cfg = config_from_dimensionless(rho_r=float(rho_r), phi=phis_batch, alpha=alpha,
                                    theta=thetas_batch, N=N, M=M, B=B, seed=seed, p=p,
                                    replenish=True, device=str(select_device(device)))
    sim = BatchedSwarm(cfg)
    out = sim.run(T)
    t0 = int((1.0 - tail) * T)
    active = out["active_count"][:, t0:]
    has = active > 0
    hden = has.sum(dim=1).clamp(min=1)
    def wmean(key):
        return ((out[key][:, t0:] * has).sum(dim=1) / hden)
    return {
        "sigma": wmean("sigma"),
        "z_local": wmean("z_local"),
        "z_global": out["z_global"][:, t0:].mean(dim=1),
        "active_frac": active.mean(dim=1) / N,
        "phi_expl": out["exploring_fraction"][:, t0:].mean(dim=1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zmin", type=float, default=2.0)
    ap.add_argument("--zmax", type=float, default=25.0)
    ap.add_argument("--nz", type=int, default=10)
    ap.add_argument("--theta-min", type=float, default=0.05)
    ap.add_argument("--theta-max", type=float, default=0.5)
    ap.add_argument("--n-theta", type=int, default=10)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.05, 0.15, 0.3, 0.5])
    ap.add_argument("--R", type=int, default=16, help="replicas per cell")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--tail", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=1.9, help="near-Brownian random walk")
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="phase_grid")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.nz, args.n_theta, args.R, args.T, args.phis = 5, 5, 6, 400, [0.05, 0.3]

    z_grid = np.linspace(args.zmin, args.zmax, args.nz)
    rho_grid = z_grid / (math.pi * args.p)
    theta_grid = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    phi_grid = np.array(args.phis, dtype=float)
    n_theta, n_phi, R = args.n_theta, len(phi_grid), args.R

    # fold (theta, phi) into the batch: order = [theta, phi, replica]
    th_mesh, ph_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")  # (n_theta, n_phi)
    thetas_batch = torch.tensor(np.repeat(th_mesh.flatten(), R), dtype=torch.float32)
    phis_batch = torch.tensor(np.repeat(ph_mesh.flatten(), R), dtype=torch.float32)

    fields = ["sigma", "z_local", "z_global", "active_frac", "phi_expl"]
    grids = {f: np.zeros((args.nz, n_theta, n_phi)) for f in fields}

    print(f"phase grid: {args.nz} rho_r x {n_theta} theta x {n_phi} phi x R={R}, "
          f"N={args.N}, T={args.T}, alpha={args.alpha} (random walk), replenish=True", flush=True)
    for i, rho in enumerate(rho_grid):
        red = run_rho(rho, thetas_batch, phis_batch, alpha=args.alpha, N=args.N, M=args.M,
                      T=args.T, p=args.p, tail=args.tail, device=args.device, seed=args.seed + i)
        for f in fields:
            v = red[f].reshape(n_theta, n_phi, R).mean(dim=2).cpu().numpy()  # (n_theta, n_phi)
            grids[f][i] = v
        sig = grids["sigma"][i]
        print(f"  z~{z_grid[i]:5.1f} (rho_r={rho:.2f})  sigma[min..max]={sig.min():.2f}..{sig.max():.2f}  "
              f"active_frac[max]={grids['active_frac'][i].max():.3f}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, f"{args.tag}.csv")
    with open(csv, "w") as fh:
        fh.write("z_set,rho_r,theta,phi," + ",".join(fields) + "\n")
        for i, rho in enumerate(rho_grid):
            for j, th in enumerate(theta_grid):
                for k, ph in enumerate(phi_grid):
                    vals = [f"{grids[f][i, j, k]:.6g}" for f in fields]
                    fh.write(f"{z_grid[i]:.6g},{rho:.6g},{th:.6g},{ph:.6g}," + ",".join(vals) + "\n")
    np.savez(os.path.join(args.outdir, f"{args.tag}.npz"), z=z_grid, rho=rho_grid,
             theta=theta_grid, phi=phi_grid, **grids)
    print(f"wrote {csv}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        extent = [theta_grid.min(), theta_grid.max(), z_grid.min(), z_grid.max()]
        fig, axes = plt.subplots(1, n_phi, figsize=(4.3 * n_phi, 4.2), squeeze=False)
        vmax = np.nanpercentile(grids["sigma"], 98)
        th_c, z_c = np.meshgrid(theta_grid, z_grid)
        for k in range(n_phi):
            ax = axes[0][k]
            Z = grids["sigma"][:, :, k]
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap="viridis",
                           vmin=0, vmax=max(vmax, 1.2))
            ax.contour(th_c, z_c, Z, levels=[1.0], colors="r", linewidths=2)
            ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"mean degree $z\ (\propto\rho_r)$")
            ax.set_title(f"$\\phi={phi_grid[k]:.2f}$")
            fig.colorbar(im, ax=ax)
        fig.suptitle(r"Branching ratio $\sigma$ over $(\rho_r,\theta)$ per $\phi$; red = $\sigma=1$")
        fig.tight_layout()
        png = os.path.join(args.outdir, f"{args.tag}.png")
        fig.savefig(png, dpi=130); print(f"wrote {png}", flush=True)
    except ImportError:
        print("matplotlib unavailable; skipped plot", flush=True)


if __name__ == "__main__":
    main()
