#!/usr/bin/env python3
"""
Patch-lifecycle test (paper limit i): does the swarm LEAVE a patch before it is empty (marginal-value
/ Charnov signature), and is the leaving caused by BLOCKING rather than by the patch simply running
out? Closes the "blocked <=> worked-out" causal gap the paper had flagged as asserted-not-shown.

For each patch (source slot), from spawn to depletion, we track:
  * occupancy  -- agents within the interaction radius r of the source
  * remaining resource (source_cap), normalised by capacity
and, per tick, how released (active->inactive) agents split into:
  * BLOCKED   -- still have active neighbours but a/k < theta (squeezed out by crowding)
  * ISOLATED  -- no active neighbours left (their local patch dissolved)

Measured:
  * giving-up density GUD = remaining resource when occupancy falls to half its peak. GUD > 0 means
    patches are abandoned BEFORE depletion (MVT-consistent); GUD ~ 0 means they are drained.
  * released-by-blocking fraction = reverted_blocked / (reverted_blocked + reverted_isolated).
  * mean occupancy as a function of remaining resource (the build-up-then-disperse lifecycle curve).

CPU only. Usage: python3 patch_lifecycle_test.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def run(args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, rho_r=args.rho_r, phi=args.phi,
                              theta=args.theta, step=args.step, pull=args.pull,
                              consumption=args.consumption, capacity=args.capacity,
                              converge=True, seed=args.seed, device="cpu",
                              levy_alpha=args.levy, exploit_step=args.exploit_step)
    sim = MobileCascade(cfg)
    r = float(sim.r)
    T = args.T
    occ = np.zeros((T, args.B, args.K), dtype=np.float32)
    cap = np.zeros((T, args.B, args.K), dtype=np.float32)
    rev_b = np.zeros(T); rev_i = np.zeros(T)                    # released-by-blocking / -isolation
    for t in range(T):
        sim.step()
        d = sim._pair_dist(sim.pos, sim.sources)               # (B,N,K)
        occ[t] = (d <= r).sum(dim=1).cpu().numpy()             # agents within r of each source
        cap[t] = sim.source_cap.cpu().numpy()
        rev_b[t] = float(sim._reverted_blocked.sum())
        rev_i[t] = float(sim._reverted_isolated.sum())

    # --- per-patch lifecycles: segment each (b,k) time series at respawns (cap jumps up) ---
    cap_frac = cap / args.capacity
    gud, peak_occ, drained = [], [], 0
    occ_by_res = [[] for _ in range(args.nbins)]               # occupancy binned by remaining-resource
    t0 = int(args.warm * T)
    for b in range(args.B):
        for k in range(args.K):
            c = cap_frac[:, b, k]; o = occ[:, b, k]
            # segment starts where cap increases (respawn) or at t0
            starts = [t0] + [t for t in range(t0 + 1, T) if c[t] > c[t - 1] + 1e-6]
            starts = sorted(set(starts))
            for si, s in enumerate(starts):
                e = starts[si + 1] if si + 1 < len(starts) else T
                seg_o = o[s:e]; seg_c = c[s:e]
                if len(seg_o) < args.min_life or seg_o.max() < args.min_peak:
                    continue                                    # not a real, occupied patch life
                pk = seg_o.max(); pk_t = int(seg_o.argmax())
                peak_occ.append(pk)
                # dispersal point: first tick AFTER the peak where occupancy < half peak
                after = seg_o[pk_t:]
                below = np.where(after < 0.5 * pk)[0]
                if len(below) == 0:
                    continue
                disp = pk_t + int(below[0])
                gud.append(float(seg_c[disp]))                  # resource remaining at dispersal
                if seg_c[disp] < 0.02:
                    drained += 1
                # lifecycle curve: occupancy vs remaining resource over the segment
                for tt in range(len(seg_o)):
                    bi = min(args.nbins - 1, int(seg_c[tt] * args.nbins))
                    occ_by_res[bi].append(seg_o[tt])

    gud = np.array(gud); peak_occ = np.array(peak_occ)
    rb = rev_b.sum(); ri = rev_i.sum()
    blocked_release_frac = rb / (rb + ri + 1e-9)
    lifecycle = np.array([np.mean(x) if x else np.nan for x in occ_by_res])  # index 0..nbins-1 = low..high res
    return {
        "n_patches": len(gud),
        "gud_mean": float(np.mean(gud)) if len(gud) else np.nan,
        "gud_median": float(np.median(gud)) if len(gud) else np.nan,
        "gud_frac_gt0": float((gud > 0.05).mean()) if len(gud) else np.nan,   # abandoned with >5% left
        "frac_drained": drained / max(len(gud), 1),
        "peak_occ_mean": float(np.mean(peak_occ)) if len(peak_occ) else np.nan,
        "blocked_release_frac": float(blocked_release_frac),
        "gud": gud.astype(np.float32),
        "lifecycle_occ_vs_res": lifecycle.astype(np.float32),   # mean occupancy per remaining-resource bin
        "res_bin_centers": ((np.arange(args.nbins) + 0.5) / args.nbins).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=150); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--B", type=int, default=16); ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--warm", type=float, default=0.25)
    ap.add_argument("--rho_r", type=float, default=3.0); ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=0.20)
    ap.add_argument("--step", type=float, default=20.0); ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0); ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--levy", type=float, default=None)
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--min-life", type=int, default=6, dest="min_life")
    ap.add_argument("--min-peak", type=float, default=3.0, dest="min_peak")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="patch_lifecycle")
    args = ap.parse_args()
    mode = "Levy(a=%.2f)" % args.levy if args.levy is not None else "random walk"
    print(f"patch lifecycle: N={args.N} K={args.K} theta={args.theta} exploration={mode}", flush=True)
    r = run(args)
    print(f"  n_patches={r['n_patches']}  peak_occ={r['peak_occ_mean']:.1f}", flush=True)
    print(f"  GUD (resource left at dispersal): mean={r['gud_mean']:.2f} median={r['gud_median']:.2f} "
          f"| abandoned with >5%% left: {100*r['gud_frac_gt0']:.0f}%%  | drained (~0): {100*r['frac_drained']:.0f}%%",
          flush=True)
    print(f"  released-by-BLOCKING fraction = {r['blocked_release_frac']:.2f} "
          f"(vs by-isolation {1-r['blocked_release_frac']:.2f})", flush=True)
    print(f"  lifecycle occ vs remaining-resource (low->high): "
          f"{np.array2string(r['lifecycle_occ_vs_res'], precision=1)}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, **{k: v for k, v in r.items()})
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
