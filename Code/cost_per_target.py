#!/usr/bin/env python3
"""
Cost-per-job efficiency:  bits / target consumed   and   steps / target consumed.  LOWER = BETTER.

Why not harvest/cost? A fixed-horizon ratio rewards inactivity: a swarm that barely moves and barely
talks spends almost nothing, so "harvest per metre" can look excellent while no work happens. That
degeneracy has bitten this project twice (the old "oil per message" maximised at the no-communication
corner; and "heterogeneity improves movement efficiency" where agents simply parked and harvest fell
38%). Fixing the JOB and measuring the COST removes it: an idle swarm never consumes the targets, so
its cost per target diverges.

Definitions (all dimensionless):
  targets = cumulative consumed / capacity          (one unit = one full patch eaten)
  steps   = cumulative distance / step_scale        (distance in units of one exploration step)
  bits    = beacon_msgs*bits_beacon + activation_msgs*bits_activation
            (reported separately too: the no-recruit baseline also pays beacons, so the MARGINAL
             communication cost of recruitment is the activation channel)

Measured at a FIXED JOB SIZE: the cumulative cost at the tick when cumulative targets first reach X,
with X chosen so every condition in a cell reaches it (else the cell is reported censored).

Saves sweeps/cost_per_target.npz.   Usage: python3 cost_per_target.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def trajectory(theta, phi, patchiness, args):
    """Cumulative (targets, steps, beacon_msgs, activation_msgs) per tick, averaged over replicas."""
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, phi=phi,
                              theta=theta, capacity=args.capacity, consumption=args.consumption,
                              levy_alpha=args.levy, clump_count=args.clumps,
                              clump_radius=patchiness * args.L, step=args.step, pull=args.pull,
                              seed=args.seed)
    sim = MobileCascade(cfg)
    out = sim.run(args.T)
    tgt = out["consumed"].mean(0).cumsum(0) / args.capacity          # patch-equivalents eaten
    stp = out["distance"].mean(0).cumsum(0) / args.step              # step-equivalents travelled
    bec = out["msgs_beacon"].mean(0).cumsum(0)
    act = out["msgs_activation"].mean(0).cumsum(0)
    return (tgt.cpu().numpy(), stp.cpu().numpy(), bec.cpu().numpy(), act.cpu().numpy())


def cost_at_job(traj, X):
    """Cumulative costs at the first tick where cumulative targets >= X. None if never reached."""
    tgt, stp, bec, act = traj
    idx = np.searchsorted(tgt, X)
    if idx >= len(tgt):
        return None
    return dict(steps=float(stp[idx]), beacon=float(bec[idx]), activation=float(act[idx]),
                ticks=int(idx + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--rho_r", type=float, default=3.0)
    ap.add_argument("--theta-lo", type=float, default=0.20, dest="theta_lo")
    ap.add_argument("--theta-base", type=float, default=0.99, dest="theta_base")
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--clumps", type=int, default=3)
    ap.add_argument("--patchiness", type=float, nargs="+", default=[0.025, 0.075, 0.19, 0.5])
    ap.add_argument("--phis", type=float, nargs="+", default=[0.005, 0.02, 0.08])
    ap.add_argument("--capacity", type=float, default=200.0)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--levy", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--bits_beacon", type=float, default=1.0)
    ap.add_argument("--bits_activation", type=float, default=1.0)
    ap.add_argument("--job_frac", type=float, default=0.8,
                    help="job size X = job_frac * min(targets reached by any condition in the cell)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="cost_per_target")
    args = ap.parse_args()

    patch = np.array(args.patchiness); phis = np.array(args.phis)
    sh = (len(phis), len(patch))
    G = {k: np.full(sh, np.nan) for k in
         ["steps_r", "steps_b", "act_r", "act_b", "bits_r", "bits_b",
          "steps_ratio", "act_ratio", "bits_ratio", "job", "exchange"]}
    print(f"cost per target (LOWER=better): steps/target and bits/target at a FIXED job size\n"
          f"  rho_r={args.rho_r} clumps={args.clumps} K={args.K} N={args.N} T={args.T}", flush=True)
    for i, phi in enumerate(phis):
        for j, p in enumerate(patch):
            tr = trajectory(args.theta_lo, phi, p, args)          # recruiting
            tb = trajectory(args.theta_base, phi, p, args)        # no-recruitment baseline
            X = args.job_frac * min(tr[0][-1], tb[0][-1])         # a job both conditions complete
            cr, cb = cost_at_job(tr, X), cost_at_job(tb, X)
            if cr is None or cb is None or X <= 0:
                print(f"  phi={phi:.3f} patch={p:.3f}: CENSORED (job not reached)", flush=True); continue
            sr, sb = cr["steps"] / X, cb["steps"] / X             # steps per target
            ar, ab = cr["activation"] / X, cb["activation"] / X   # activation msgs per target
            br = (cr["beacon"] * args.bits_beacon + cr["activation"] * args.bits_activation) / X
            bb = (cb["beacon"] * args.bits_beacon + cb["activation"] * args.bits_activation) / X
            G["steps_r"][i, j], G["steps_b"][i, j] = sr, sb
            G["act_r"][i, j], G["act_b"][i, j] = ar, ab
            G["bits_r"][i, j], G["bits_b"][i, j] = br, bb
            G["steps_ratio"][i, j] = sb / max(sr, 1e-12)          # >1 => recruiting is CHEAPER
            G["act_ratio"][i, j] = ab / max(ar, 1e-12)
            G["bits_ratio"][i, j] = bb / max(br, 1e-12)
            G["job"][i, j] = X
            # EXCHANGE RATE: steps of movement saved per EXTRA activation message spent.
            # This compares the two incommensurate costs without summing them, and is exactly the
            # break-even lambda*: recruiting pays iff you value one message at fewer than this
            # many steps. (Negative extra-comms => recruiting is a pure win on both axes.)
            d_steps, d_msgs = (sb - sr), (ar - ab)
            G["exchange"][i, j] = d_steps / d_msgs if d_msgs > 1e-12 else np.inf
            print(f"  phi={phi:.3f} patch={p:.3f} | job={X:6.1f} targets | "
                  f"steps/target r/b = {sr:8.1f}/{sb:8.1f} (x{sb/sr:5.2f} cheaper) | "
                  f"act-msgs/target r/b = {ar:8.1f}/{ab:8.1f} (x{ab/max(ar,1e-12):5.2f}) | "
                  f"bits/target r/b = {br:8.1f}/{bb:8.1f} | EXCHANGE {G['exchange'][i,j]:6.2f} steps/msg", flush=True)
    # ---- the two efficiencies are incommensurate (metres vs bits): never summed. Report where each
    #      is optimised, and the Pareto set where recruiting wins on BOTH. ----
    SR, AR = G["steps_ratio"], G["act_ratio"]        # >1 => recruiting cheaper on that axis
    def cell(idx): return f"phi={phis[idx[0]]:.3f}, patch={patch[idx[1]]:.3f}"
    print("\n=== where does recruitment pay? (ratio = baseline cost / recruit cost; >1 = cheaper) ===",
          flush=True)
    if np.isfinite(SR).any():
        i = np.unravel_index(np.nanargmax(SR), SR.shape)
        print(f"  best MOVEMENT saving : {cell(i)}  steps x{SR[i]:.2f} cheaper", flush=True)
    if np.isfinite(AR).any():
        j = np.unravel_index(np.nanargmax(AR), AR.shape)
        print(f"  best COMMS saving    : {cell(j)}  activation-msgs x{AR[j]:.2f} cheaper", flush=True)
    both = [(a, b) for a in range(SR.shape[0]) for b in range(SR.shape[1])
            if np.isfinite(SR[a, b]) and np.isfinite(AR[a, b]) and SR[a, b] > 1 and AR[a, b] > 1]
    print(f"  cells where recruiting is cheaper on BOTH axes: {len(both)}/{SR.size}", flush=True)
    for a, b in both:
        print(f"     {cell((a,b))}: steps x{SR[a,b]:.2f}, comms x{AR[a,b]:.2f}", flush=True)
    # Pareto set (non-dominated when maximising both savings)
    pts = [(a, b) for a in range(SR.shape[0]) for b in range(SR.shape[1])
           if np.isfinite(SR[a, b]) and np.isfinite(AR[a, b])]
    par = [q for q in pts if not any((SR[r] >= SR[q] and AR[r] >= AR[q]) and
                                     (SR[r] > SR[q] or AR[r] > AR[q]) for r in pts)]
    print(f"  Pareto-optimal cells (can't improve one saving without losing the other):", flush=True)
    for q in par:
        print(f"     {cell(q)}: steps x{SR[q]:.2f}, comms x{AR[q]:.2f}", flush=True)

    EX = G["exchange"]
    if np.isfinite(EX).any():
        k = np.unravel_index(np.nanargmax(np.where(np.isfinite(EX), EX, -np.inf)), EX.shape)
        print(f"  BEST EXCHANGE RATE   : {cell(k)}  {EX[k]:.2f} steps saved per extra message", flush=True)
        print(f"  (break-even lambda*: recruiting pays iff 1 message is worth < this many steps)", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, patchiness=patch, phis=phis, **G)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
