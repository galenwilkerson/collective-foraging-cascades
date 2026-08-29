#!/usr/bin/env python3
"""
Efficiency phase-diagram sweep over (rho_r, theta) at fixed (alpha, phi).

Goal (Galen, 2026-07-04): map the space and find the *efficient* operating
points -- where do we get efficient communication, efficient total movement
(work), and the best oil-cleaned-per-cost -- before touching the more abstract
SOC / RL story.

Method (matches the paper's fixed-horizon philosophy, section 6/8):
  * Run every simulation for a fixed horizon T; record full (B,T) trajectories;
    derive all cost/efficiency measures downstream. No early stopping.
  * theta is folded into the batch dimension (config_from_dimensionless takes a
    (B,) tensor), rho_r is looped outside. The batch also carries n_rep Monte
    Carlo replicas per theta, so B = n_theta * n_rep. Replicas are independent
    because each of the B runs draws its own random initial state.

Per (rho_r, theta) cell we reduce the trajectories to, and average over the
replicas (keeping the across-replica std as the fluctuation):
    frac_cleaned      (oil0 - oilT) / oil0
    cleaned           oil0 - oilT                     (absolute)
    distance          sum_t locomotion                (total movement / "work")
    msgs_node         sum_t per-node broadcasts       (communication)
    msgs_edge         sum_t per-edge messages         (communication)
    time_to_clean     first t with oil <= eps*oil0    (else censored at T)
    eff_move          cleaned / distance              (oil per unit movement)
    eff_node          cleaned / msgs_node             (oil per node-message)
    eff_edge          cleaned / msgs_edge             (oil per edge-message)

Outputs (in --outdir):
    sweep_<tag>.csv    tidy long format, one row per (rho_r, theta)
    sweep_<tag>.npz    2D arrays [n_rho, n_theta] for each measure + axes
    *.png              heatmaps (if matplotlib present and --plot)

Usage:
    python3 sweep_efficiency.py                 # full default sweep
    python3 sweep_efficiency.py --quick         # tiny smoke test
    python3 sweep_efficiency.py --plot          # also write heatmaps
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# allow running from anywhere: import the model sitting next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_swarm import BatchedSwarm, config_from_dimensionless, select_device  # noqa: E402


# summary fields reduced from each (B,) per-run vector
_FIELDS = [
    "frac_cleaned", "cleaned", "distance", "msgs_node", "msgs_edge",
    "time_to_clean",
    # --- rate observables (the discriminating signal; completeness saturates) ---
    "k_decay",     # depletion-corrected decay-rate constant 1/tau (two-point log slope)
    "peak_rate",   # max oil removed in a single tick
    "mean_rate",   # cleaned / time_to_clean (average active-phase rate)
    # --- yields (oil cleaned per unit cost) ---
    "eff_move", "eff_node", "eff_edge",
]

# heavy-tailed under the alpha<=1 Levy tail -> summarize by median, not mean
_HEAVY = {"distance", "eff_move"}


def _reduce_cell(out, oil_init, T, eps):
    """Reduce one batch of (B,T) trajectories to a dict of per-run (B,) vectors."""
    oilT = out["oil_remaining"][:, -1]
    cleaned = (oil_init - oilT).clamp(min=0.0)
    frac = cleaned / oil_init.clamp(min=1e-9)

    dist = out["distance"].sum(dim=1)
    mnode = out["msgs_node"].sum(dim=1)
    medge = out["msgs_edge"].sum(dim=1)

    thresh = (eps * oil_init).unsqueeze(1)                     # (B,1)
    below = out["oil_remaining"] <= thresh                     # (B,T)
    ever = below.any(dim=1)
    first = below.float().argmax(dim=1).to(oil_init.dtype)     # first True, or 0 if none
    ttc = torch.where(ever, first, torch.full_like(oil_init, float(T)))

    # --- rate observables ---------------------------------------------------
    # We care about how FAST oil is cleaned, not just how much (completeness
    # saturates at long T). Raw rate falls trivially as oil depletes, so the
    # cleanest measure is the depletion-corrected decay-rate CONSTANT
    # k = 1/tau, from a two-point log slope between the 80% and 20% crossings.
    O = out["oil_remaining"]                                   # (B,T)
    dO = (O[:, :-1] - O[:, 1:]).clamp(min=0.0)                 # oil removed / tick
    first_drop = (oil_init - O[:, 0]).clamp(min=0.0).unsqueeze(1)
    peak_rate = torch.cat([first_drop, dO], dim=1).max(dim=1).values

    mean_rate = cleaned / ttc.clamp(min=1.0)

    hi = O <= (0.8 * oil_init).unsqueeze(1)
    lo = O <= (0.2 * oil_init).unsqueeze(1)
    ever_hi, ever_lo = hi.any(dim=1), lo.any(dim=1)
    t_hi = hi.float().argmax(dim=1).to(oil_init.dtype)
    t_lo = lo.float().argmax(dim=1).to(oil_init.dtype)
    dt = t_lo - t_hi
    valid = ever_hi & ever_lo & (dt > 0)
    k_decay = torch.where(valid, math.log(0.8 / 0.2) / dt.clamp(min=1.0),
                          torch.zeros_like(oil_init))

    return {
        "frac_cleaned": frac,
        "cleaned": cleaned,
        "distance": dist,
        "msgs_node": mnode,
        "msgs_edge": medge,
        "time_to_clean": ttc,
        "k_decay": k_decay,
        "peak_rate": peak_rate,
        "mean_rate": mean_rate,
        "eff_move": cleaned / dist.clamp(min=1e-9),
        "eff_node": cleaned / mnode.clamp(min=1e-9),
        "eff_edge": cleaned / medge.clamp(min=1e-9),
        "_censored": (~ever).to(oil_init.dtype),               # book-keeping
    }


def run_sweep(rho_list, theta_list, *, n_rep, T, N, M, phi, alpha,
              device=None, seed=0, eps=0.05, verbose=True):
    dev = select_device(device)
    n_rho, n_theta = len(rho_list), len(theta_list)
    theta_t = torch.tensor(theta_list, dtype=torch.float32)
    thetas_batch = torch.repeat_interleave(theta_t, n_rep)     # (n_theta*n_rep,)
    B = n_theta * n_rep

    # results[field] -> (n_rho, n_theta) mean/std/median over replicas
    mean = {f: np.zeros((n_rho, n_theta)) for f in _FIELDS}
    std = {f: np.zeros((n_rho, n_theta)) for f in _FIELDS}
    med = {f: np.zeros((n_rho, n_theta)) for f in _FIELDS}
    censored = np.zeros((n_rho, n_theta))
    finite_frac = np.zeros((n_rho, n_theta))   # fraction of replicas with finite distance

    t0 = time.time()
    for i, rho in enumerate(rho_list):
        cfg = config_from_dimensionless(
            rho_r=float(rho), phi=phi, alpha=alpha, theta=thetas_batch,
            N=N, M=M, B=B, seed=seed + i, device=str(dev),
        )
        sim = BatchedSwarm(cfg)
        oil_init = sim.blob_radius.sum(dim=1).clone()          # (B,) true t=0 oil
        out = sim.run(T)
        red = _reduce_cell(out, oil_init, T, eps)

        for f in _FIELDS:
            v = red[f].reshape(n_theta, n_rep)                 # rows=theta, cols=replica
            mean[f][i] = v.mean(dim=1).cpu().numpy()
            std[f][i] = v.std(dim=1).cpu().numpy()
            med[f][i] = v.median(dim=1).values.cpu().numpy()   # robust to heavy Levy tail
        censored[i] = red["_censored"].reshape(n_theta, n_rep).mean(dim=1).cpu().numpy()
        finite_frac[i] = torch.isfinite(red["distance"]).reshape(
            n_theta, n_rep).float().mean(dim=1).cpu().numpy()

        if verbose:
            kd = mean["k_decay"][i]
            jbest = int(kd.argmax())
            print(f"[{i+1:>2}/{n_rho}] rho_r={rho:6.3f}  "
                  f"k_decay(peak@theta={theta_list[jbest]:.3f})={kd.max():.4g}  "
                  f"mean_rate~{mean['mean_rate'][i].mean():.3g}  "
                  f"dist_finite={finite_frac[i].mean():.2f}  "
                  f"[{time.time()-t0:5.1f}s]", flush=True)

    return {
        "rho": np.asarray(rho_list, dtype=float),
        "theta": np.asarray(theta_list, dtype=float),
        "mean": mean, "std": std, "median": med,
        "censored": censored, "finite_frac": finite_frac,
        "meta": dict(n_rep=n_rep, T=T, N=N, M=M, phi=phi, alpha=alpha,
                     eps=eps, seed=seed, device=str(dev), B=B),
    }


def save_csv(res, path):
    rho, theta = res["rho"], res["theta"]
    cols = ["rho_r", "theta"]
    for f in _FIELDS:
        cols += [f + "_mean", f + "_std", f + "_median"]
    cols += ["frac_censored", "dist_finite_frac"]
    lines = [",".join(cols)]
    for i, r in enumerate(rho):
        for j, th in enumerate(theta):
            row = [f"{r:.6g}", f"{th:.6g}"]
            for f in _FIELDS:
                row += [f"{res['mean'][f][i, j]:.6g}",
                        f"{res['std'][f][i, j]:.6g}",
                        f"{res['median'][f][i, j]:.6g}"]
            row += [f"{res['censored'][i, j]:.6g}", f"{res['finite_frac'][i, j]:.6g}"]
            lines.append(",".join(row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def save_npz(res, path):
    kw = {"rho": res["rho"], "theta": res["theta"],
          "censored": res["censored"], "finite_frac": res["finite_frac"]}
    for f in _FIELDS:
        kw[f + "_mean"] = res["mean"][f]
        kw[f + "_std"] = res["std"][f]
        kw[f + "_median"] = res["median"][f]
    np.savez(path, **kw)


def save_heatmaps(res, outdir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping heatmaps", flush=True)
        return
    rho, theta = res["rho"], res["theta"]
    extent = [theta.min(), theta.max(), rho.min(), rho.max()]
    # rate observables first (the live signal), then yields and cost
    panels = ["k_decay", "mean_rate", "peak_rate",
              "eff_node", "eff_edge", "distance", "time_to_clean", "frac_cleaned"]
    for f in panels:
        # heavy-tailed fields plotted by median; everything else by mean
        Z = res["median"][f] if f in _HEAVY else res["mean"][f]  # (n_rho, n_theta)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis")
        ax.set_xlabel(r"$\theta$ (cascade threshold)")
        ax.set_ylabel(r"$\rho_r$ (connectivity density)")
        ax.set_title(f + (" (median)" if f in _HEAVY else ""))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        p = os.path.join(outdir, f"{tag}_{f}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
    print(f"wrote {len(panels)} heatmaps to {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho-min", type=float, default=0.5)
    ap.add_argument("--rho-max", type=float, default=8.0)
    ap.add_argument("--n-rho", type=int, default=15)
    ap.add_argument("--theta-min", type=float, default=0.05)
    ap.add_argument("--theta-max", type=float, default=0.5)
    ap.add_argument("--n-theta", type=int, default=15)
    ap.add_argument("--n-rep", type=int, default=16, help="MC replicas per cell")
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05,
                    help="oil <= eps*oil0 counts as cleaned (time_to_clean)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test grid")
    args = ap.parse_args()

    if args.quick:
        args.n_rho, args.n_theta, args.n_rep, args.T = 4, 4, 4, 80

    rho_list = np.linspace(args.rho_min, args.rho_max, args.n_rho).tolist()
    theta_list = np.linspace(args.theta_min, args.theta_max, args.n_theta).tolist()

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or ("quick" if args.quick else
                       f"rho{args.n_rho}xth{args.n_theta}_rep{args.n_rep}_T{args.T}")

    print(f"sweep: {args.n_rho} rho_r x {args.n_theta} theta x {args.n_rep} rep, "
          f"T={args.T}, N={args.N}, M={args.M}, alpha={args.alpha}, phi={args.phi}",
          flush=True)

    res = run_sweep(rho_list, theta_list, n_rep=args.n_rep, T=args.T, N=args.N,
                    M=args.M, phi=args.phi, alpha=args.alpha, device=args.device,
                    seed=args.seed, eps=args.eps)

    csv_path = os.path.join(args.outdir, f"sweep_{tag}.csv")
    npz_path = os.path.join(args.outdir, f"sweep_{tag}.npz")
    save_csv(res, csv_path)
    save_npz(res, npz_path)
    print(f"wrote {csv_path}\nwrote {npz_path}", flush=True)
    if args.plot:
        save_heatmaps(res, args.outdir, f"sweep_{tag}")


if __name__ == "__main__":
    main()
