#!/usr/bin/env python3
"""Cascade FSS at MATCHED criticality (sigma_local=1 per field, per N) -- the decisive branching-class test.

The fixed-theta sweep confounded the universality test: the operating point sigma drifted with field and N.
Here we CALIBRATE theta per (field,N) so sigma_aval~1 (true criticality), then measure the cascade-size
exponent. Question: at matched sigma=1, do uniform and clumped give the SAME tau?
  - same tau (=3/2)  -> universality holds; the field is an IRRELEVANT perturbation (earlier gap = sigma drift).
  - stable tau != 3/2 for clumped -> a genuinely different branching class (then explain via offspring tail).

Theta calibration = bisection (sigma decreases monotonically with theta), short runs; final measurement at the
calibrated theta with full stats. GPU, vectorized. Output -> /mnt/newssd/.../fss3_<field>_N<N>.npz.
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_branching_test import run_setting   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
NS = [50, 100, 200, 400, 800]
FIELDS = ["uniform", "scalefree"]


def make_args(N, field, B, T):
    clump = (dict(clump_count=6, clump_radius=40.0, clump_beta=1.5) if field == "scalefree"
             else dict(clump_count=None, clump_radius=40.0, clump_beta=None))
    return argparse.Namespace(B=B, N=N, K=max(2, N // 5), L=800.0, rho_r=3.0, phi=0.05, step=20.0,
                              pull=0.1, consumption=1.0, capacity=40.0, seed=0, device="cuda",
                              levy=1.5, exploit_step=3.0, tail=0.5, T=T, big=8, **clump)


def calibrate_theta(field, N, lo=0.03, hi=0.38, tol=0.025, iters=8):
    """Bisect theta so sigma_aval ~ 1 (sigma monotonically DECREASES with theta)."""
    a = make_args(N, field, B=20, T=350)
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = run_setting(mid, a)["sigma_aval"]
        if best is None or abs(s - 1.0) < abs(best[1] - 1.0):
            best = (mid, s)
        if abs(s - 1.0) < tol:
            return mid, s
        if s > 1.0:
            lo = mid          # sigma too high -> raise theta
        else:
            hi = mid
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'field':10} {'N':>5} {'theta*':>7} {'sigma':>7} {'tauCCDF':>8} {'maxsize':>8} {'n':>7}", flush=True)
    for field in FIELDS:
        for N in NS:
            th, scal = calibrate_theta(field, N)
            a = make_args(N, field, B=32, T=700)
            r = run_setting(th, a)
            np.savez(os.path.join(OUT, f"fss3_{field}_N{N}.npz"),
                     N=np.array([N]), theta=np.array([th]), sigma_aval=np.array([r["sigma_aval"]]),
                     sigma_lo=np.array([r["sigma_aval_lo"]]), sigma_hi=np.array([r["sigma_aval_hi"]]),
                     tau=np.array([r["tau"]]), sizes=r["sizes"].astype(np.float32),
                     per_aval_sigma=r["per_aval_sigma"], local_deg=np.array([r["local_deg_mean"]]))
            print(f"{field:10} {N:5d} {th:7.3f} {r['sigma_aval']:7.3f} {r['tau']:8.2f} "
                  f"{int(r['sizes'].max()):8d} {len(r['sizes']):7d}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
