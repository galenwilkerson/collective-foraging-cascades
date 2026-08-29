#!/usr/bin/env python3
"""
Ping economy: how much communication can be cut before the cascade degrades?

Galen's idea: both channels are emitted every tick, which is almost certainly wasteful.
  * "I am here" (beacon) exists only to maintain the proximity graph. The graph decorrelates on
    ~(r/step)^alpha ticks -- for near-Brownian motion that is TENS of ticks, so beaconing every tick
    is mostly re-sending unchanged information.
  * "I found a target" (activation) is naturally an EVENT. Re-announcing it every tick while still
    active is redundant.

THE RISK, and why this is not free: the rule is degree-normalised, a/k >= theta. A stale graph
corrupts k, and missing neighbours mean SMALLER k, hence EASIER activation. Staleness therefore biases
toward OVER-RECRUITMENT -- exactly the failure mode blocking exists to prevent. So the question is not
"can we send less" but "how much less before the cascade destabilises", and the active fraction is the
thing to watch.

Sweeps beacon interval (and event-driven beaconing) x alpha, reporting the message saving against the
damage: active fraction, harvest, and cost per target.

Usage: python3 ping_economy.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def run(args, alpha, interval=1, move_frac=0.0, event=False):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, p=args.p,
                              phi=args.phi, theta=args.theta, step=args.step,
                              exploit_step=args.ars * args.step, pull=args.pull, levy_alpha=alpha,
                              clump_count=args.clumps, clump_radius=args.patchiness * args.L,
                              capacity=args.cap, consumption=args.cons,
                              beacon_interval=interval, beacon_move_frac=move_frac,
                              activation_event_only=event, seed=args.seed)
    sim = MobileCascade(cfg)
    o = sim.run(args.T)
    tg = (o["consumed"].mean(0).cumsum(0) / args.cap).cpu().numpy()
    st = (o["distance"].mean(0).cumsum(0) / args.step).cpu().numpy()
    be = o["msgs_beacon"].mean(0).cumsum(0).cpu().numpy()
    ac = o["msgs_activation"].mean(0).cumsum(0).cpu().numpy()
    X = args.job_frac * args.K
    i = int(np.searchsorted(tg, X))
    cens = i >= len(tg)
    j = min(i, len(tg) - 1)
    den = X if not cens else max(tg[-1], 1e-9)
    return dict(active=o["active_count"].mean().item() / args.N,
                casc=o["cascade_size"].mean().item() / args.N,
                harvest=float(tg[-1]), censored=cens,
                steps=float(st[j] / den), bits=float((be[j] + ac[j]) / den),
                beacon=float(be[j] / den), act=float(ac[j] / den),
                r_over_step=sim.r / args.step)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--N", type=int, default=103)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--rho_r", type=float, default=3.41)
    ap.add_argument("--p", type=float, default=0.614)
    ap.add_argument("--phi", type=float, default=0.007)
    ap.add_argument("--patchiness", type=float, default=0.082)
    ap.add_argument("--clumps", type=int, default=6)
    ap.add_argument("--ars", type=float, default=0.051)
    ap.add_argument("--pull", type=float, default=0.336)
    ap.add_argument("--theta", type=float, default=0.217)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--cons", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--job_frac", type=float, default=0.25)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.9, 1.4, 1.9])
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--movefracs", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ref = run(args, args.alphas[0])
    print(f"r/step = {ref['r_over_step']:.1f}  ->  graph decorrelation time ~ (r/step)^alpha:")
    for a in args.alphas:
        print(f"    alpha={a}: tau ~ {ref['r_over_step']**a:.0f} ticks")
    print()

    for a in args.alphas:
        base = run(args, a)
        print(f"=== alpha={a}  (baseline: active {base['active']:.2f}, bits/tgt {base['bits']:.0f}, "
              f"steps/tgt {base['steps']:.1f}, harvest {base['harvest']:.1f}) ===")
        print("   FIXED beacon interval")
        for k in args.intervals:
            r = run(args, a, interval=k)
            d_act = 100 * (r["active"] / max(base["active"], 1e-9) - 1)
            d_h = 100 * (r["harvest"] / max(base["harvest"], 1e-9) - 1)
            flag = "  <-- OVER-RECRUITING" if d_act > 15 else ("  (censored)" if r["censored"] else "")
            print(f"     every {k:2d} ticks: bits/tgt {r['bits']:7.1f} "
                  f"({100*(1-r['bits']/max(base['bits'],1e-9)):+5.1f}%)  active {r['active']:.2f} "
                  f"({d_act:+5.1f}%)  harvest {d_h:+6.1f}%{flag}")
        print("   EVENT-DRIVEN beacon (re-broadcast after moving x*r)")
        for mf in args.movefracs:
            r = run(args, a, move_frac=mf)
            d_act = 100 * (r["active"] / max(base["active"], 1e-9) - 1)
            d_h = 100 * (r["harvest"] / max(base["harvest"], 1e-9) - 1)
            flag = "  <-- OVER-RECRUITING" if d_act > 15 else ""
            print(f"     move>{mf:4.2f}r  : bits/tgt {r['bits']:7.1f} "
                  f"({100*(1-r['bits']/max(base['bits'],1e-9)):+5.1f}%)  active {r['active']:.2f} "
                  f"({d_act:+5.1f}%)  harvest {d_h:+6.1f}%{flag}")
        e = run(args, a, event=True)
        print(f"   EVENT-ONLY activation: bits/tgt {e['bits']:7.1f} "
              f"({100*(1-e['bits']/max(base['bits'],1e-9)):+5.1f}%)  "
              f"activation/tgt {e['act']:.1f} vs {base['act']:.1f} "
              f"({100*(1-e['act']/max(base['act'],1e-9)):+.0f}%)  harvest "
              f"{100*(e['harvest']/max(base['harvest'],1e-9)-1):+.1f}%")
        both = run(args, a, move_frac=args.movefracs[1], event=True)
        print(f"   BOTH (event beacon {args.movefracs[1]}r + event activation): bits/tgt "
              f"{both['bits']:7.1f} ({100*(1-both['bits']/max(base['bits'],1e-9)):+5.1f}%)  "
              f"active {both['active']:.2f}  harvest "
              f"{100*(both['harvest']/max(base['harvest'],1e-9)-1):+.1f}%")
        print()


if __name__ == "__main__":
    main()
