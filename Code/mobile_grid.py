#!/usr/bin/env python3
"""
Thorough grid search of the minimal depleting mobile-cascade model over the dimensionless
controls (rho_r x theta x phi), R replicas per cell. Records steady-state, SOC-relevant
observables so we can see WHERE the SOC cycle and critical bursts occur:

    active_frac_mean   mean activity
    active_frac_cov    temporal coeff. of variation of activity  (cycle amplitude; high = it oscillates)
    sigma_mean         mean branching ratio
    sigma_p95          95th-percentile branching  (do bursts reach/exceed criticality, sigma>=1 ?)
    cascade_mean       mean avalanche size
    cascade_p95        95th-percentile avalanche size  (heavy tail?)

Coupled model (converge=True). Each cell is a batch of R replicas on the GPU.

Usage:
    python3 mobile_grid.py            # coarse grid -> sweeps/mobile_grid.{npz,png}
    python3 mobile_grid.py --quick
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def run_cell(rho_r, theta, phi, step, args):
    cfg = MobileCascadeConfig(B=args.R, N=args.N, K=args.K, rho_r=rho_r, phi=phi, theta=theta,
                              step=step, pull=args.pull, consumption=args.consumption,
                              capacity=args.capacity, converge=True, seed=args.seed,
                              levy_alpha=args.levy, exploit_step=args.exploit_step,
                              clump_count=args.clumps, clump_radius=args.clump_radius)
    sim = MobileCascade(cfg)
    out = sim.run(args.T)
    t0 = int((1.0 - args.tail) * args.T); Tw = args.T - t0
    af = (out["active_count"][:, t0:] / args.N)               # (R,Tw)
    sig = out["sigma"][:, t0:]
    casc = out["cascade_size"][:, t0:]
    cov = (af.std(dim=1) / (af.mean(dim=1) + 1e-6)).mean().item()
    # --- raw cost ledgers (per agent-tick): keep separate, derive efficiencies downstream ---
    eps, kap = 1e-9, args.consumption
    U    = out["consumed"][:, t0:].sum(dim=1) / (kap * args.N * Tw)          # harvest, fraction of max
    Dh   = out["distance"][:, t0:].sum(dim=1) / (args.N * Tw * step)         # movement per agent-tick (step units)
    Mact = out["msgs_activation"][:, t0:].sum(dim=1) / (args.N * Tw)         # cascade pings per agent-tick
    Mbea = out["msgs_beacon"][:, t0:].sum(dim=1) / (args.N * Tw)             # presence pings per agent-tick (~1)
    Mtot = Mact + Mbea
    return {
        "active_frac_mean": af.mean().item(),
        "active_frac_cov": cov,
        "sigma_mean": sig.mean().item(),
        "sigma_p95": torch.quantile(sig.reshape(-1), 0.95).item(),
        "cascade_mean": casc.mean().item(),
        "cascade_p95": torch.quantile(casc.reshape(-1), 0.95).item(),
        "utilization": U.mean().item(),
        # raw ledgers (derive any efficiency downstream):
        "ledger_U": U.mean().item(),
        "ledger_move": Dh.mean().item(),
        "ledger_beacon": Mbea.mean().item(),
        "ledger_activation": Mact.mean().item(),
        # convenience efficiencies (comms = per-node broadcast pings):
        "eta_E": (U / (Dh + eps)).mean().item(),                    # harvest per movement
        "eta_C": (U / (Mact + eps)).mean().item(),                  # harvest per cascade ping
        "eta_W": (U / (Dh + Mtot + eps)).mean().item(),             # harvest per (movement + all bits, 1:1)
        "subst": (Mact / (Dh + eps)).mean().item(),                 # cascade pings per unit movement
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zmin", type=float, default=2.0)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--theta-min", type=float, default=0.05)
    ap.add_argument("--theta-max", type=float, default=0.5)
    ap.add_argument("--n-theta", type=int, default=8)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.03, 0.05, 0.1])
    ap.add_argument("--steps", type=float, nargs="+", default=None,
                    help="if set, sweep step-scale as the 3rd axis (phi fixed) -- the Levy scaling range")
    ap.add_argument("--levy", type=float, default=None, help="Levy exponent (exploration); None = random walk")
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--clumps", type=int, default=None, help="if set: sources live in this many spatial clumps (patchy)")
    ap.add_argument("--clump-radius", type=float, default=40.0, dest="clump_radius")
    ap.add_argument("--R", type=int, default=24)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--tail", type=float, default=0.4)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="mobile_grid")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.nz, args.n_theta, args.R, args.T, args.phis = 4, 4, 8, 400, [0.05]

    z_grid = np.linspace(args.zmin, args.zmax, args.nz)
    rho_grid = z_grid / (math.pi * args.p)
    theta_grid = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    # third axis: step-scale (the Levy scaling range) if --steps given, else phi
    if args.steps:
        third_name, third_grid, phi_fixed = "step", np.array(args.steps, dtype=float), float(args.phis[0])
    else:
        third_name, third_grid, phi_fixed = "phi", np.array(args.phis, dtype=float), None
    n3 = len(third_grid)
    fields = ["active_frac_mean", "active_frac_cov", "sigma_mean", "sigma_p95",
              "cascade_mean", "cascade_p95", "utilization", "eta_E", "eta_C", "eta_W", "subst",
              "ledger_U", "ledger_move", "ledger_beacon", "ledger_activation"]
    grids = {f: np.zeros((args.nz, args.n_theta, n3)) for f in fields}
    base_fields = ["ledger_U", "ledger_move", "ledger_beacon", "ledger_activation", "eta_W"]
    base = {f: np.zeros((args.nz, n3)) for f in base_fields}

    def cell_ps(v):   # -> (phi, step) for a third-axis value v
        return (phi_fixed if third_name == "step" else float(v)), (float(v) if third_name == "step" else args.step)

    print(f"mobile grid: {args.nz} rho_r x {args.n_theta} theta x {n3} {third_name} x R={args.R}, "
          f"N={args.N}, K={args.K}, T={args.T}, levy={args.levy}", flush=True)
    for i, rho in enumerate(rho_grid):
        for j, th in enumerate(theta_grid):
            for k, v in enumerate(third_grid):
                ph, st = cell_ps(v)
                r = run_cell(float(rho), float(th), ph, st, args)
                for f in fields:
                    grids[f][i, j, k] = r[f]
        s95 = grids["sigma_p95"][i]
        print(f"  z~{z_grid[i]:5.1f}  active[max]={grids['active_frac_mean'][i].max():.2f}  "
              f"cov[max]={grids['active_frac_cov'][i].max():.2f}  sigma_p95[max]={s95.max():.2f}",
              flush=True)

    print("baseline (no recruitment, theta=0.99) ...", flush=True)
    for i, rho in enumerate(rho_grid):
        for k, v in enumerate(third_grid):
            ph, st = cell_ps(v)
            rb = run_cell(float(rho), 0.99, ph, st, args)
            for f in base_fields:
                base[f][i, k] = rb[f]

    os.makedirs(args.outdir, exist_ok=True)
    base_kw = {f"base_{f}": base[f] for f in base_fields}
    np.savez(os.path.join(args.outdir, f"{args.tag}.npz"), z=z_grid, rho=rho_grid, theta=theta_grid,
             eta_W_baseline=base["eta_W"], **base_kw, third_name=np.array([third_name]), third=third_grid,
             phi=(third_grid if third_name == "phi" else np.array([phi_fixed], dtype=float)),
             step=(third_grid if third_name == "step" else np.array([args.step], dtype=float)),
             **grids)
    print(f"wrote {os.path.join(args.outdir, args.tag)}.npz", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_phi = len(third_grid)
        panels = ["active_frac_cov", "sigma_p95", "eta_W", "subst"]
        extent = [theta_grid.min(), theta_grid.max(), z_grid.min(), z_grid.max()]
        fig, axes = plt.subplots(len(panels), n_phi, figsize=(4.2 * n_phi, 3.6 * len(panels)),
                                 squeeze=False)
        for pr, f in enumerate(panels):
            for k in range(n_phi):
                ax = axes[pr][k]
                Z = grids[f][:, :, k]
                im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap="viridis")
                if f == "sigma_p95":
                    th_c, z_c = np.meshgrid(theta_grid, z_grid)
                    ax.contour(th_c, z_c, Z, levels=[1.0], colors="r", linewidths=2)
                ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$z\ (\propto\rho_r)$")
                ax.set_title(f"{f}, {third_name}={third_grid[k]:.2f}")
                fig.colorbar(im, ax=ax)
        fig.suptitle("Minimal depleting mobile-cascade grid (red = $\\sigma_{p95}=1$)")
        fig.tight_layout()
        png = os.path.join(args.outdir, f"{args.tag}.png")
        fig.savefig(png, dpi=120); print(f"wrote {png}", flush=True)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
