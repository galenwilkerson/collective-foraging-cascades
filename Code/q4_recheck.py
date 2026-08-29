#!/usr/bin/env python3
"""
Q4 RE-CHECK: does "recruitment is efficient iff resources are SPARSE and CLUMPED" survive?

The paper's Q4 claim (main.tex l.697-700, Fig 1 RHS) was measured under two conditions the global
Sobol sweep has since called into question:

 (a) EXPLORATION REGIME.  It was run at levy_alpha = 1.0. Stage A found alpha is the single strongest
     driver of movement cost, spanning ~1000x, and alpha=1.0 sits in the second-WORST bin (median
     579 steps/target vs 94 at alpha~1.8). The baseline (no recruitment) relies entirely on
     exploration, so handicapping exploration may be what made recruitment look good.

 (b) THE MEANING OF "CLUMPED".  It was operationalised via (patchiness, clump_count) varied
     independently. Those do not isolate clustering: 20.5% of the Stage-A sample had
     cover = n*pi*(R/L)^2 > 1, i.e. patches tiling the arena -- nominally "clumped", physically
     UNIFORM. Marginal patchiness curves came out flat as a result. The clustering signal actually
     lives in SOURCES PER CLUMP (K/n: steps 271->184, bits 596->435), which was never an axis.

So this sweep re-derives the Q4 map in corrected coordinates:
    cover           = clump_count * pi * (clump_radius/L)^2   -- CONSTRAINED < 1 so clustered means clustered
    src_per_clump   = K / clump_count                          -- the coordinate the signal lives in
    phi                                                        -- sparse -> abundant (the "sparse" half of the claim)
    levy_alpha in {1.0, 1.5, 2.0}                              -- was frozen at 1.0

Each cell is run against R random CONTEXTS for the remaining axes (drawn from the same Sobol space)
and reduced by median, so the map is marginalised over the rest of phase space rather than pinned to
one corner -- the mistake that produced the retracted heterogeneity result.

Metric: cost to consume a fixed job, recruiting (theta_lo) vs no-recruitment baseline (theta=0.99).
Reported as ratio baseline/recruit; >1 means recruiting is CHEAPER on that axis. Movement and
communication are kept separate and never summed.

Usage: python3 q4_recheck.py
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
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--covers", type=float, nargs="+", default=[0.01, 0.05, 0.2, 0.6])
    ap.add_argument("--spc", type=float, nargs="+", default=[1.0, 3.0, 8.0, 20.0])
    ap.add_argument("--phis", type=float, nargs="+", default=[0.004, 0.02, 0.10])
    ap.add_argument("--reps", type=int, default=8, help="random contexts per cell")
    ap.add_argument("--theta_lo", type=float, default=0.20)
    ap.add_argument("--theta_base", type=float, default=0.99)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--bits_beacon", type=float, default=1.0)
    ap.add_argument("--bits_activation", type=float, default=1.0)
    ap.add_argument("--job_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sweeps", "q4_recheck.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ctx = qmc.Sobol(d=len(AXES), scramble=True, seed=args.seed).random(args.reps)
    cells = [(a, c, s, f) for a in args.alphas for c in args.covers
             for s in args.spc for f in args.phis]
    total = len(cells) * args.reps * 2
    print(f"Q4 re-check: {len(cells)} cells x {args.reps} contexts x 2 conditions = {total} runs\n"
          f"  coords: cover(<1) x src_per_clump x phi x alpha; median over contexts\n"
          f"  out = {args.out}", flush=True)

    rows, t0, n = [], time.time(), 0
    for (a, cov, spc, phi) in cells:
        for ci, u in enumerate(ctx):
            P = unit_to_params(u)
            P["levy_alpha"], P["phi"] = float(a), float(phi)
            # invert the corrected clustering coords back to the model's raw knobs
            nclump = max(1, int(round(P["K"] / spc)))
            P["clump_count"] = nclump
            P["patchiness"] = float(np.sqrt(cov / (nclump * np.pi)))
            if P["patchiness"] > 0.7:            # keep patches inside the arena
                continue
            for cond, th in (("recruit", args.theta_lo), ("baseline", args.theta_base)):
                Q = dict(P); Q["theta"] = th
                if cond == "baseline":
                    Q["s_rel"] = 0.0             # baseline never recruits; spread is meaningless
                try:
                    r = run_point(Q, args)
                    r.update(cell_alpha=a, cell_cover=cov, cell_spc=spc, cell_phi=phi,
                             cond=cond, ctx=ci)
                    rows.append(r)
                except Exception as e:
                    print(f"  FAILED a={a} cov={cov} spc={spc} phi={phi} {cond}: "
                          f"{type(e).__name__}: {e}", flush=True)
                n += 1
        if len(rows) and n % 192 < 2:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            el = time.time() - t0
            print(f"  [{n}/{total}] {el/60:.1f} min, ~{el/max(n,1)*(total-n)/60:.0f} min left",
                  flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
