#!/usr/bin/env python3
"""
Q2 --- SOC self-tuning test (go/no-go on the homeostasis thesis).

Run steady-state (replenishing) dynamic episodes across a range of GLOBAL densities
z_global (set via connectivity), and ask whether the LOCAL observables self-tune to
marginal values INDEPENDENT of z_global:

    z_local  (mean RGG degree experienced by active agents) -> z_c (~4.5) ?
    sigma    (cascade branching ratio)                      -> 1 ?
    phi      (exploring fraction)                           -> phi* (stable) ?

Decision:
  * PASS (self-organization): z_local_ss is ~FLAT in z_global (pinned near z_c), not tracking it.
  * FAIL: z_local_ss tracks z_global (the diagonal) -> no self-tuning, homeostasis thesis wrong.

Exploration uses near-Brownian steps (alpha=1.9) to match the paper's unbiased-random-walk
framing (Levy is out of scope). Uses the replenishing-resource mode for a stationary test.

Usage:
    python3 q2_self_tuning.py            # default sweep -> sweeps/q2_self_tuning.csv + .png
    python3 q2_self_tuning.py --quick
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_swarm import BatchedSwarm, config_from_dimensionless, select_device  # noqa: E402

Z_C = 4.51


@torch.no_grad()
def run_point(z_global, *, phi, theta, alpha, N, M, B, T, p, tail, device, seed):
    """One steady-state episode; return steady-state (active-weighted) local observables."""
    rho_r = z_global / (math.pi * p)
    cfg = config_from_dimensionless(rho_r=rho_r, phi=phi, alpha=alpha, theta=theta,
                                    N=N, M=M, B=B, seed=seed, p=p, replenish=True,
                                    device=str(select_device(device)))
    sim = BatchedSwarm(cfg)
    out = sim.run(T)
    t0 = int((1.0 - tail) * T)
    active = out["active_count"][:, t0:]                 # (B, Tw)
    has = active > 0                                     # ticks with a cascade
    def wmean(key):
        w = out[key][:, t0:]
        num = (w * has).sum()
        den = has.sum().clamp(min=1)
        return (num / den).item()
    zloc = wmean("z_local")
    sig = wmean("sigma")
    phi_expl = out["exploring_fraction"][:, t0:].mean().item()
    zg = out["z_global"][:, t0:].mean().item()
    active_frac = active.mean().item() / N
    return dict(z_set=z_global, z_global=zg, z_local=zloc, sigma=sig,
                phi=phi_expl, active_frac=active_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zmin", type=float, default=2.0)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--nz", type=int, default=10)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--tail", type=float, default=0.4, help="steady-state window (last fraction)")
    ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=0.18)
    ap.add_argument("--alpha", type=float, default=1.9, help="near-Brownian (random-walk framing)")
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="initial-condition arm")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="q2_self_tuning")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.nz, args.N, args.B, args.T, args.seeds = 5, 60, 32, 400, [0]

    z_grid = np.linspace(args.zmin, args.zmax, args.nz).tolist()
    print(f"Q2 self-tuning: {args.nz} z_global x seeds={args.seeds}, N={args.N}, T={args.T}, "
          f"theta={args.theta}, replenish=True", flush=True)
    rows = []
    for s in args.seeds:
        for z in z_grid:
            r = run_point(z, phi=args.phi, theta=args.theta, alpha=args.alpha, N=args.N,
                          M=args.M, B=args.B, T=args.T, p=args.p, tail=args.tail,
                          device=args.device, seed=s)
            r["seed"] = s
            rows.append(r)
            print(f"  seed={s} z_global={r['z_global']:6.2f}  z_local={r['z_local']:6.2f}  "
                  f"sigma={r['sigma']:.3f}  phi={r['phi']:.3f}  active_frac={r['active_frac']:.3f}",
                  flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, f"{args.tag}.csv")
    cols = ["seed", "z_set", "z_global", "z_local", "sigma", "phi", "active_frac"]
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" for c in cols) + "\n")
    print(f"wrote {csv}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for s in args.seeds:
            d = [r for r in rows if r["seed"] == s]
            zg = np.array([r["z_global"] for r in d])
            a1.plot(zg, [r["z_local"] for r in d], "o-", label=f"seed {s}")
            a2.plot(zg, [r["sigma"] for r in d], "o-", label=f"seed {s}")
        lo, hi = min(r["z_global"] for r in rows), max(r["z_global"] for r in rows)
        a1.plot([lo, hi], [lo, hi], "k--", lw=1, label=r"$z_{\rm local}=z_{\rm global}$ (no self-tuning)")
        a1.axhline(Z_C, color="C3", ls=":", lw=1.5, label=r"$z_c\approx4.5$")
        a1.set_xlabel(r"$z_{\rm global}$ (imposed)"); a1.set_ylabel(r"$z_{\rm local}$ (active agents)")
        a1.set_title("Q2: does local degree self-tune?"); a1.legend(fontsize=8)
        a2.axhline(1.0, color="C3", ls=":", lw=1.5, label=r"$\sigma=1$ (marginal)")
        a2.set_xlabel(r"$z_{\rm global}$ (imposed)"); a2.set_ylabel(r"$\sigma$ (branching ratio)")
        a2.set_title("Q2: branching ratio"); a2.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(args.outdir, f"{args.tag}.png")
        fig.savefig(png, dpi=130)
        print(f"wrote {png}", flush=True)
    except ImportError:
        print("matplotlib unavailable; skipped plot", flush=True)


if __name__ == "__main__":
    main()
