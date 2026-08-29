#!/usr/bin/env python3
"""
Does the swarm ADAPTIVELY regulate exploration against resource state? (the emergent-RL test)

The learning_curve result said the configuration carries resource information but plateaus =
homeostatic tracking, not learning. This asks the complementary question: is the explore/exploit
BALANCE regulated by resource availability, and if so is it (a) an adaptive feedback loop or (b) a
mechanical fallback (less resource -> fewer seeds -> less recruitment -> more agents idle-exploring,
with no regulation at all)?

Discriminator = temporal structure of exploring_fraction vs total remaining resource:
  * sign of the correlation: negative (explore rises as resource falls) is necessary but not
    sufficient -- the mechanical fallback also predicts it.
  * LAG: does exploration lead resource depletion (anticipatory -> regulation) or lag it (reactive ->
    mechanical)?
  * PHASE PORTRAIT: a closed loop / hysteresis in the (resource, explore) plane is a feedback cycle
    (relaxation oscillation); a monotone line with no area is a passive one-to-one map.
  * ABLATION: converge=False cuts the recruitment->motion coupling. If the regulation is a real
    feedback loop it should weaken/vanish; if it is pure seed-accounting it should survive unchanged.

CPU only. Usage: python3 adaptive_exploration.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def run(converge, T=1200, seed=0):
    """Return PER-BATCH time series (B, T-burn). Batch-averaging the series would smear out the
    relaxation oscillation (independent phases cancel), so all statistics are computed per trajectory
    and averaged afterwards."""
    cfg = MobileCascadeConfig(B=32, N=120, K=15, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                              levy_alpha=1.0, clump_count=6, clump_radius=40.0, capacity=40.0,
                              consumption=1.0, step=20.0, pull=0.12, converge=converge,
                              device="cpu", seed=seed)
    sim = MobileCascade(cfg)
    out = sim.run(T)
    b0 = 200
    res = out["resource"].cpu().numpy()[:, b0:]              # (B, T)
    exp = out["exploring_fraction"].cpu().numpy()[:, b0:]
    return res, exp


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def xcorr(a, b, maxlag=40):
    """corr(a(t), b(t+lag)) for lag in [-maxlag, maxlag]. Positive lag => b lags a."""
    a, b = zscore(a), zscore(b)
    n = len(a)
    lags = np.arange(-maxlag, maxlag + 1)
    out = []
    for L in lags:
        if L < 0:
            c = np.corrcoef(a[-L:], b[:n + L])[0, 1]
        elif L > 0:
            c = np.corrcoef(a[:n - L], b[L:])[0, 1]
        else:
            c = np.corrcoef(a, b)[0, 1]
        out.append(c)
    return lags, np.array(out)


def loop_area(res, exp):
    """Signed area enclosed by the (resource, explore) trajectory (shoelace), normalised.
    Nonzero area = hysteresis = a feedback loop rather than a one-to-one map."""
    x, y = zscore(res), zscore(exp)
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / len(x)
    return area


def main():
    print("adaptive exploration vs resource depletion (CPU, N=120, B=32, T=1200, burn-in 200)")
    print("all statistics computed PER trajectory then averaged (batch-mean would smear the cycle)\n")
    for label, conv in [("recruiting (coupling ON) ", True), ("ablated  (coupling OFF)", False)]:
        RES, EXP = run(conv)
        c0s, lags_peak, cpeaks, areas = [], [], [], []
        lag_grid = None
        cstack = []
        for res, exp in zip(RES, EXP):
            if exp.std() < 1e-6:               # saturated trajectory: no oscillation to analyse
                continue
            lags, c = xcorr(res, exp); lag_grid = lags
            cstack.append(c)
            c0s.append(c[len(c) // 2])
            i0 = np.argmin(c); lags_peak.append(lags[i0]); cpeaks.append(c[i0])
            areas.append(loop_area(res, exp))
        if not cstack:
            print(f"{label}: all trajectories saturated (mean explore {EXP.mean():.2f}) — no cycle.\n")
            continue
        cmean = np.mean(cstack, 0)
        i0 = np.argmin(cmean); peak_lag = lag_grid[i0]
        print(f"{label}:")
        print(f"   corr(resource, explore) lag 0 = {np.mean(c0s):+.3f} ± {np.std(c0s):.3f}   "
              f"(POSITIVE => explore high when resource abundant-undiscovered; NEG => rises with scarcity)")
        print(f"   averaged-xcorr strongest {cmean[i0]:+.3f} at lag {peak_lag:+d} ticks "
              f"({'explore LEADS' if peak_lag>0 else 'explore LAGS' if peak_lag<0 else 'simultaneous'})")
        print(f"   per-traj phase-loop area = {np.mean(areas):.3f} ± {np.std(areas):.3f}  "
              f"({'HYSTERESIS -> feedback cycle' if np.mean(areas)>0.05 else 'thin -> ~one-to-one'})")
        print(f"   mean explore frac = {EXP.mean():.2f}  (within-traj osc std {EXP.std(1).mean():.3f})")
        print()

    print("Reading:")
    print("  * fat per-traj loop = a genuine explore/exploit feedback CYCLE (regulation, not a static map)")
    print("  * ablation collapsing the cycle = the coupling CAUSES the regulation")
    print("  * but no accumulation over time (learning_curve) => homeostatic SOC regulation, NOT RL")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps", "adaptive_exploration.npz")
    RES_on, EXP_on = run(True); RES_off, EXP_off = run(False)
    np.savez(out, res_on=RES_on, exp_on=EXP_on, res_off=RES_off, exp_off=EXP_off)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
