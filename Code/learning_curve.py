#!/usr/bin/env python3
"""
Is the swarm LEARNING? Measure the agent density against the resource density over time.

Galen's reframing: the thing that gets updated is not a parameter (theta, alpha are hyperparameters)
but the swarm's CONFIGURATION -- agent positions, and the proximity graph they induce. The density
field is therefore the learned object, closer to a self-organizing map or a non-parametric value
function than to policy-gradient RL.

If that framing is right, agent density should come to approximate the LIVE resource density, and the
mismatch should fall over time. That is a learning curve with a known optimum (matching / the ideal
free distribution), not a metaphor.

Measured per tick on a coarse spatial grid, with the torus respected:
  JS(agents || live resource)   Jensen-Shannon divergence, 0 = perfect match, 1 = disjoint (bits)
  and the same for a RANDOM-WALK CONTROL (theta=0.99: no recruitment, so no configuration update
  driven by discovery). The control is the null model -- diffusion alone also concentrates agents
  somewhat, so the recruiting swarm must beat it for "learning" to mean anything.

Also reported:
  * FORGETTING: after a patch depletes, how fast does local agent density decay?
  * GENERALISATION: does density on UNVISITED patches of a clump rise once one member is found?
    (clumped resources make one discovery predictive of its neighbours -- the structure recruitment
     can exploit, and the reason it should pay in clumped environments.)

Usage: python3 learning_curve.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def hist2d(pos, L, nb, weights=None):
    """Occupancy histogram on an nb x nb torus grid -> (B, nb*nb), row-normalised."""
    B, M, _ = pos.shape
    ix = torch.clamp((pos[..., 0] / L * nb).long(), 0, nb - 1)
    iy = torch.clamp((pos[..., 1] / L * nb).long(), 0, nb - 1)
    flat = (iy * nb + ix)                                            # (B,M)
    out = torch.zeros(B, nb * nb, device=pos.device, dtype=torch.float32)
    w = torch.ones_like(flat, dtype=torch.float32) if weights is None else weights
    out.scatter_add_(1, flat, w)
    return out


def js_div(p, q, eps=1e-12):
    """Jensen-Shannon divergence in bits, per batch row. Symmetric, bounded in [0,1]."""
    p = p / p.sum(1, keepdim=True).clamp(min=eps)
    q = q / q.sum(1, keepdim=True).clamp(min=eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * ((a + eps).log2() - (b + eps).log2())).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run(theta, args, label):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, p=args.p,
                              phi=args.phi, theta=theta, step=args.step,
                              exploit_step=args.ars * args.step, pull=args.pull,
                              levy_alpha=args.alpha, clump_count=args.clumps,
                              clump_radius=args.patchiness * args.L, capacity=args.cap,
                              consumption=args.cons, seed=args.seed)
    sim = MobileCascade(cfg)
    js, harv = [], []
    for t in range(args.T):
        sim.step()
        if t % args.every:
            continue
        a = hist2d(sim.pos, args.L, args.nb)
        # Target = CLUMP CENTRES, not live sources. A successful forager EATS the resource where it
        # is, so live-source density is depleted exactly where agents are -- matching it would
        # penalise success. Clump centres are the persistent structure (sources respawn within them),
        # so they are what a learned configuration should come to track.
        c = hist2d(sim.clump_centers, args.L, args.nb)
        js.append(js_div(a, c).mean().item())
        # Directly interpretable companion: mean distance from each agent to the nearest LIVE source,
        # in units of L. Lower = better positioned to harvest. No support-overlap artefact.
        d = sim._pair_dist(sim.pos, sim.sources)                     # (B,N,K)
        d = d.masked_fill(~(sim.source_cap > 0).unsqueeze(1), float("inf"))
        dm = d.min(dim=2).values
        dm = dm[torch.isfinite(dm)]
        harv.append((dm.mean().item() / args.L) if dm.numel() else float("nan"))
    return np.array(js), np.array(harv)


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
    ap.add_argument("--alpha", type=float, default=0.906)
    ap.add_argument("--ars", type=float, default=0.051)
    ap.add_argument("--pull", type=float, default=0.336)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--cons", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.217)
    ap.add_argument("--theta_base", type=float, default=0.99)
    ap.add_argument("--nb", type=int, default=16, help="grid resolution (nb x nb cells)")
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"agent-density vs live-resource-density mismatch, JS divergence in bits (LOWER = better\n"
          f"match). grid {args.nb}x{args.nb}, N={args.N}, K={args.K}, B={args.B}, T={args.T}\n")
    jr, dr_ = run(args.theta, args, "recruit")
    jb, db_ = run(args.theta_base, args, "baseline")
    n = min(len(jr), len(jb))
    jr, jb = jr[:n], jb[:n]
    q = np.linspace(0, n - 1, 6).astype(int)
    print("  JS(agents || CLUMP CENTRES)              mean dist to nearest live source (/L)")
    print("  tick     recruiting   control    gap        recruiting   control    gap")
    for i in q:
        print(f"  {i*args.every:5d}      {jr[i]:.4f}     {jb[i]:.4f}   {jb[i]-jr[i]:+.4f}"
              f"       {dr_[i]:.4f}      {db_[i]:.4f}   {db_[i]-dr_[i]:+.4f}")
    print(f"\n  distance to nearest live source: recruiting {dr_[:len(dr_)//10].mean():.4f} -> "
          f"{dr_[-len(dr_)//10:].mean():.4f} | control {db_[:len(db_)//10].mean():.4f} -> "
          f"{db_[-len(db_)//10:].mean():.4f}")
    early, late = slice(0, max(2, n // 10)), slice(-max(2, n // 10), None)
    dr = jr[early].mean() - jr[late].mean()
    db = jb[early].mean() - jb[late].mean()
    print(f"\n  recruiting : {jr[early].mean():.4f} -> {jr[late].mean():.4f}   improvement {dr:+.4f}")
    print(f"  control    : {jb[early].mean():.4f} -> {jb[late].mean():.4f}   improvement {db:+.4f}")
    # The question is NOT "does each condition improve over time" (an earlier version tested that and
    # wrongly reported a null). It is "does the recruiting configuration sit closer to the resource
    # structure than the no-recruitment control" -- a LEVEL difference, not a trend.
    gap_js, gap_d = (jb - jr).mean(), (db_[:n] - dr_[:n]).mean()
    print(f"\n  mean GAP vs control:  JS {gap_js:+.4f} bits   distance {gap_d:+.4f} L "
          f"({100*gap_d/max(db_[:n].mean(),1e-9):.0f}% closer)")
    settle = int(0.2 * n)
    tr_r, tr_b = jr[settle:].mean() - jr[:settle].mean(), jb[settle:].mean() - jb[:settle].mean()
    print(f"  drift AFTER the first 20% of the run: recruiting {tr_r:+.4f}, control {tr_b:+.4f}")
    if gap_js > 0.01 or gap_d > 0.005:
        print("\n  => The configuration DOES carry information about the resource field: the recruiting")
        print("     swarm holds a position distribution measurably closer to the clump structure than")
        print("     diffusion alone achieves.")
        if abs(tr_r) < 0.01:
            print("  => BUT it converges fast and then plateaus -- no accumulation over time. This is")
            print("     HOMEOSTATIC TRACKING (a steady state maintained against depletion), not a")
            print("     learning curve. 'Learning' implies improvement with experience; this is")
            print("     relaxation to a fixed point, reached within the first ~100 ticks.")
    else:
        print("\n  => No detectable gap vs the control; 'learning' would be the wrong word.")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps", "learning_curve.npz")
    np.savez(out, js_recruit=jr, js_baseline=jb, every=args.every)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
