#!/usr/bin/env python3
"""
Analyse the global phase-space sweep: which knobs drive cost-per-job, and which INTERACT with
threshold spread?

Two questions, two instruments:

 (1) WHICH AXES MATTER.  A gradient-boosted surrogate is fit to each outcome and permutation
     importance is read off it. This is censoring-tolerant (censored points are modelled as their own
     binary outcome rather than dropped), which classical Sobol index estimators are not.

 (2) DOES SPREAD HELP OR HURT -- AND WHERE.  This is the actual question, and it is a question about
     INTERACTION, not main effect. For each other axis j the sample is split into tertiles of j, and
     within each tertile the effect of threshold spread is measured directly (median cost of the
     high-spread third minus the low-spread third). If that effect changes sign across tertiles of j,
     then j MODERATES heterogeneity -- and reporting a single global "spread helps/hurts" number is
     meaningless. That is precisely the error the earlier single-operating-point sweep made.

Usage:  python3 phase_space_analyze.py [--csv sweeps/phase_space.csv]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

AXES = ["theta", "s_rel", "dist_cat", "rho_r", "p", "phi", "K", "patchiness", "clump_count",
        "levy_alpha", "ars_ratio", "pull", "deplete_ticks", "N"]
OUTCOMES = ["steps_per_target", "bits_per_target", "ticks_per_target"]


def importances(df, y, clf=False):
    X = df[AXES].values
    M = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(
        max_iter=300, random_state=0).fit(X, y)
    r = permutation_importance(M, X, y, n_repeats=8, random_state=0)
    return pd.Series(r.importances_mean, index=AXES).sort_values(ascending=False), M


def spread_effect(sub, out):
    """Median cost change from the low-spread third to the high-spread third (negative = spread HELPS)."""
    lo, hi = sub["s_rel"].quantile(1 / 3), sub["s_rel"].quantile(2 / 3)
    a, b = sub[sub.s_rel <= lo][out], sub[sub.s_rel >= hi][out]
    if len(a) < 8 or len(b) < 8:
        return np.nan
    return 100.0 * (b.median() / a.median() - 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "sweeps", "phase_space.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    fin = df[(df.censored == 0)].replace([np.inf, -np.inf], np.nan).dropna(subset=OUTCOMES)
    print(f"{len(df)} points ({df.censored.mean():.1%} censored); {len(fin)} usable\n")

    print("=== (1) which axes drive cost? (permutation importance) ===")
    for out in OUTCOMES:
        imp, _ = importances(fin, np.log10(fin[out].clip(lower=1e-9)))
        print(f"  {out}: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(6).items()))
    imp, _ = importances(df, df.censored.values, clf=True)
    print(f"  P(job never finished): " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(6).items()))

    print("\n=== (2) does threshold spread help or hurt? (%% change, high-spread vs low-spread) ===")
    print("    negative = spread REDUCES cost = heterogeneity HELPS\n")
    for out in ["steps_per_target", "bits_per_target"]:
        g = spread_effect(fin, out)
        print(f"  {out}: GLOBAL {g:+.1f}%")
        rows = []
        for ax in AXES:
            if ax == "s_rel":
                continue
            q = fin[ax].quantile([1 / 3, 2 / 3]).values
            eff = [spread_effect(s, out) for s in
                   (fin[fin[ax] <= q[0]], fin[(fin[ax] > q[0]) & (fin[ax] <= q[1])], fin[fin[ax] > q[1]])]
            if np.all(np.isfinite(eff)):
                rows.append((ax, eff, np.nanmax(eff) - np.nanmin(eff),
                             np.nanmin(eff) < 0 < np.nanmax(eff)))
        rows.sort(key=lambda r: -r[2])
        for ax, eff, rng, flip in rows:
            mark = "  <-- SIGN FLIPS" if flip else ""
            print(f"     by {ax:14s} low/mid/high = {eff[0]:+7.1f}% {eff[1]:+7.1f}% {eff[2]:+7.1f}%"
                  f"   (range {rng:5.1f}pp){mark}")
        print()

    print("=== (3) best operating points found (lowest cost, uncensored) ===")
    for out in ["steps_per_target", "bits_per_target"]:
        r = fin.nsmallest(3, out)
        print(f"  {out}:")
        for _, x in r.iterrows():
            print(f"    {x[out]:9.2f} | theta={x.theta:.2f} s_rel={x.s_rel:.2f} rho_r={x.rho_r:.1f} "
                  f"phi={x.phi:.3f} patch={x.patchiness:.3f} alpha={x.levy_alpha:.2f} "
                  f"ars={x.ars_ratio:.2f} pull={x.pull:.2f}")
    # Pareto front: cheap on BOTH movement and communication (the two are incommensurate, never summed)
    S, Bv = fin["steps_per_target"].values, fin["bits_per_target"].values
    par = [i for i in range(len(fin)) if not np.any((S <= S[i]) & (Bv <= Bv[i]) &
                                                    ((S < S[i]) | (Bv < Bv[i])))]
    print(f"\n=== (4) Pareto front (steps vs bits, both minimised): {len(par)} of {len(fin)} points ===")
    pf = fin.iloc[par].nsmallest(8, "steps_per_target")
    for _, x in pf.iterrows():
        print(f"    steps={x.steps_per_target:8.2f} bits={x.bits_per_target:9.2f} | theta={x.theta:.2f} "
              f"s_rel={x.s_rel:.2f} rho_r={x.rho_r:5.1f} patch={x.patchiness:.3f} ars={x.ars_ratio:.2f}")


if __name__ == "__main__":
    main()
