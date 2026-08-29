#!/usr/bin/env python3
"""
STAGE B: dense (levy_alpha x threshold-spread) phase diagram -- the one interaction Stage A supports.

Stage A (4096 Sobol points) found that threshold heterogeneity has NO detectable global effect on
cost-per-job, but that its effect on movement cost is MONOTONE in the Levy exponent: -73% at
alpha~0.55 decaying to ~0 by alpha~1.8. Only the lowest alpha bin was individually significant, and 48
tertile tests were run, so this needs targeted confirmation rather than a victory lap. That is what
this script is.

DESIGN NOTE -- why this is not just another slice. Fixing the other 12 axes at "sensible" values is
exactly the mistake that produced the original single-operating-point claim. Instead each (alpha,
s_rel) cell is evaluated against R independent random CONTEXTS drawn from the same Sobol space, and
the cell statistic is the median over contexts. So the surface reported is the effect of (alpha,
s_rel) MARGINALISED over the rest of phase space, not conditioned on one arbitrary corner of it.

Usage: python3 stage_b_alpha_spread.py --na 16 --ns 12 --reps 16
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_space_sweep import AXES, unit_to_params, run_point  # noqa: E402

AX = [a[0] for a in AXES]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--na", type=int, default=16, help="levy_alpha grid points")
    ap.add_argument("--ns", type=int, default=12, help="s_rel grid points")
    ap.add_argument("--reps", type=int, default=16, help="random contexts per cell (marginalisation)")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--bits_beacon", type=float, default=1.0)
    ap.add_argument("--bits_activation", type=float, default=1.0)
    ap.add_argument("--job_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sweeps", "stage_b_alpha_spread.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ia, isr = AX.index("levy_alpha"), AX.index("s_rel")
    alphas = np.linspace(0.40, 2.00, args.na)
    srels = np.linspace(0.00, 1.20, args.ns)
    ctx = qmc.Sobol(d=len(AXES), scramble=True, seed=args.seed).random(args.reps)

    total = args.na * args.ns * args.reps
    print(f"Stage B: {args.na} alpha x {args.ns} spread x {args.reps} random contexts = {total} runs\n"
          f"  cell statistic = median over contexts (marginalised, not a fixed slice)\n"
          f"  out = {args.out}", flush=True)
    rows, t0, n = [], time.time(), 0
    for a in alphas:
        for s in srels:
            for c, u in enumerate(ctx):
                P = unit_to_params(u)
                P["levy_alpha"], P["s_rel"] = float(a), float(s)   # pin the 2 axes, randomise the rest
                try:
                    r = run_point(P, args); r["ctx"] = c
                    rows.append(r)
                except Exception as e:
                    print(f"  FAILED a={a:.2f} s={s:.2f} ctx={c}: {type(e).__name__}: {e}", flush=True)
                n += 1
            if n % 192 == 0:
                pd.DataFrame(rows).to_csv(args.out, index=False)
                el = time.time() - t0
                print(f"  [{n}/{total}] {el/60:.1f} min, ~{el/n*(total-n)/60:.0f} min left", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
