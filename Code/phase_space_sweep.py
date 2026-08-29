#!/usr/bin/env python3
"""
GLOBAL phase-space sweep: cost-per-job across ALL model axes simultaneously.

Why not a grid? The model exposes 14 dimensionless controls. A 4-level full factorial is 4^14 ~ 268M
cells -- impossible -- and even a 2-level screen is 2^14 = 16384, more than the whole sample budget.
One-factor-at-a-time around a baseline is exactly the mistake that produced
the earlier "heterogeneity lowers efficiency" claim: it was measured at a SINGLE operating point
(rho_r=2.1, phi=0.02, patchiness=0.05, alpha=1, ARS ratio=0.15), and the effect of threshold spread
already flips sign along theta, so it will flip along the frozen axes too.

Instead: a space-filling Sobol sample over all axes at once. Every point is a different corner of the
phase space, so a surrogate fit afterwards recovers which knobs matter, and -- the actual question --
which knobs INTERACT with threshold spread. Dense 2-D phase diagrams are then drawn only on the pairs
that turn out to matter (stage B), rather than guessed at in advance.

THE JOB (fixed, and comparable at every parameter setting): consume job_frac * K patch-equivalents,
i.e. "eat this fraction of the world's resource". Cost is reported per target, so it normalises across
K. A run that never finishes is recorded as CENSORED with the fraction of the job it did complete --
that failure is signal, not missing data, and it is why a censoring-tolerant surrogate is used here
rather than classical Sobol index estimators (which would silently drop those points).

Outcomes (all dimensionless, LOWER = better): steps/target, bits/target (split beacon vs activation),
ticks/target, plus the movement phase split and the blocking/active mechanism variables.

Usage:  python3 phase_space_sweep.py --n 4096 --out sweeps/phase_space.csv
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402

# name -> (low, high, scale, kind).  'log' spans orders of magnitude; 'int' rounds; 'cat' is binary.
AXES = [
    ("theta",         0.05,  0.45, "lin", "f"),   # mean threshold
    ("s_rel",         0.00,  1.20, "lin", "f"),   # THE TREATMENT: spread = s_rel * theta (relative)
    ("dist_cat",      0.00,  1.00, "lin", "c"),   # 0 = gaussian, 1 = uniform
    ("rho_r",         0.50, 20.00, "log", "f"),   # connectivity density  (blocking lives here)
    ("p",             0.25,  1.00, "lin", "f"),   # edge probability      (z = pi*p*rho_r)
    ("phi",           0.002, 0.15, "log", "f"),   # source coverage
    ("K",             8,     64,   "log", "i"),   # number of sources
    ("patchiness",    0.01,  0.60, "log", "f"),   # clump_radius / L  -> target clumping
    ("clump_count",   1,     24,   "log", "i"),   # number of patches
    ("levy_alpha",    0.40,  2.00, "lin", "f"),   # exploration exponent (2 = Brownian)
    ("ars_ratio",     0.01,  1.00, "log", "f"),   # exploit_step/step = ARS-to-dispersal (NEVER swept before)
    ("pull",          0.02,  0.60, "log", "f"),   # convergence rate toward parents
    ("deplete_ticks", 5,     200,  "log", "f"),   # capacity/consumption = patch depletion timescale
    ("N",             60,    240,  "log", "i"),   # swarm size
]


def unit_to_params(u):
    """Map a point in [0,1]^d to the model's dimensionless controls."""
    P = {}
    for v, (name, lo, hi, scale, kind) in zip(u, AXES):
        x = lo * (hi / lo) ** v if scale == "log" else lo + (hi - lo) * v
        if kind == "i":
            x = int(round(x))
        elif kind == "c":
            x = int(x >= 0.5)
        P[name] = x
    return P


def run_point(P, args):
    spread = P["s_rel"] * P["theta"]
    cfg = MobileCascadeConfig(
        B=args.B, N=P["N"], K=P["K"], L=args.L,
        rho_r=P["rho_r"], p=P["p"], phi=P["phi"],
        theta=P["theta"],
        theta_dist=("uniform" if P["dist_cat"] else "gaussian") if spread > 0 else "const",
        theta_std=spread,
        step=args.step, exploit_step=P["ars_ratio"] * args.step, pull=P["pull"],
        levy_alpha=P["levy_alpha"],
        clump_count=P["clump_count"], clump_radius=P["patchiness"] * args.L,
        capacity=P["deplete_ticks"] * args.consumption, consumption=args.consumption,
        seed=args.seed)
    out = MobileCascade(cfg).run(args.T)
    cap = cfg.capacity

    def cum(k):
        return out[k].mean(0).cumsum(0).cpu().numpy()

    tgt = cum("consumed") / cap                     # patch-equivalents eaten
    X = args.job_frac * P["K"]                      # THE JOB: eat this fraction of the world
    i = int(np.searchsorted(tgt, X))
    censored = i >= len(tgt)
    j = min(i, len(tgt) - 1)                        # evaluate at end-of-run if censored
    denom = X if not censored else max(tgt[-1], 1e-9)

    steps = (cum("distance") / args.step)[j]
    bec, act = cum("msgs_beacon")[j], cum("msgs_activation")[j]
    m = out["active_count"].mean().item() / P["N"]
    R = dict(P)
    R.update(
        censored=int(censored), job=X, job_done_frac=float(min(tgt[-1] / X, 1.0)),
        z_nominal=float(np.pi * P["p"] * P["rho_r"]),
        steps_per_target=float(steps / denom),
        explore_per_target=float((cum("dist_explore") / args.step)[j] / denom),
        converge_per_target=float((cum("dist_converge") / args.step)[j] / denom),
        exploit_per_target=float((cum("dist_exploit") / args.step)[j] / denom),
        beacon_per_target=float(bec / denom),
        activation_per_target=float(act / denom),
        bits_per_target=float((bec * args.bits_beacon + act * args.bits_activation) / denom),
        ticks_per_target=float((j + 1) / denom),
        active_frac=float(m),
        blocked_frac=float(out["blocked"].mean().item() / P["N"]),
        cascade_size=float(out["cascade_size"].mean().item()),
        sigma_branch=float(out["sigma"].mean().item()),
    )
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4096, help="Sobol sample size (power of 2 preferred)")
    ap.add_argument("--B", type=int, default=64, help="replicas per point (batch-parallel: ~free)")
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--bits_beacon", type=float, default=1.0)
    ap.add_argument("--bits_activation", type=float, default=1.0)
    ap.add_argument("--job_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sweeps", "phase_space.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    U = qmc.Sobol(d=len(AXES), scramble=True, seed=args.seed).random(args.n)
    print(f"phase-space sweep: {args.n} Sobol points over {len(AXES)} axes, B={args.B}, T={args.T}\n"
          f"  job = consume {args.job_frac:.2f} * K patch-equivalents; cost reported per target\n"
          f"  out = {args.out}", flush=True)
    rows, t0 = [], time.time()
    for n, u in enumerate(U):
        try:
            rows.append(run_point(unit_to_params(u), args))
        except Exception as e:                       # a bad corner must not kill a 4000-point run
            print(f"  [{n}] FAILED: {type(e).__name__}: {e}", flush=True)
        if (n + 1) % 64 == 0 or n + 1 == args.n:
            pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint: resumable/inspectable
            el = time.time() - t0
            cens = np.mean([r["censored"] for r in rows]) if rows else float("nan")
            print(f"  [{n+1}/{args.n}] {el/60:.1f} min elapsed, "
                  f"~{el/(n+1)*(args.n-n-1)/60:.0f} min left, censored={cens:.1%}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(rows)} points)", flush=True)


if __name__ == "__main__":
    main()
