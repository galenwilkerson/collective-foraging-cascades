#!/usr/bin/env python3
"""
IFD test: does the swarm converge to the Ideal Free Distribution? (the formal-RL linchpin)

If the recruitment+blocking dynamics are density-dependent replicator dynamics (= the mean-field limit
of cross-learning reinforcement learning = softmax policy gradient), their rest point is the Ideal Free
Distribution: agents distribute across patches so that agents-per-patch is PROPORTIONAL to patch
richness (equivalently, per-capita intake is equalised -- no agent can do better by switching patch =
the Nash equilibrium of the habitat-selection game).

Test: give the environment HETEROGENEOUS patch richness (clump_beta -> Pareto clump sizes), let the
swarm equilibrate, then regress log(agents per clump) on log(richness per clump).
  slope = 1  -> perfect input-matching / IFD (the policy-gradient rest point)
  slope < 1  -> undermatching (rich patches under-exploited; common in real animals)
  slope = 0  -> no matching (allocation independent of richness)
Control: ablate the recruitment->motion coupling (converge=False). If the matching is the mechanism's
'policy gradient', it should collapse toward slope 0 without the coupling.

CPU only. Usage: python3 ifd_test.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


def run(converge, T=800, seed=0):
    cfg = MobileCascadeConfig(B=16, N=200, K=80, L=800.0, rho_r=2.1, phi=0.02, theta=0.18,
                              levy_alpha=1.0, clump_count=10, clump_beta=2.2, clump_radius=30.0,
                              capacity=40.0, consumption=1.0, step=20.0, pull=0.12, converge=converge,
                              device="cpu", seed=seed)
    sim = MobileCascade(cfg)
    C = sim._n_clumps
    # richness of each clump = number of sources assigned to it (fixed via _clump_idx under clump_beta)
    idx = sim._clump_idx                                   # (B, K) clump of each source
    rich = torch.zeros(sim.B, C)
    rich.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))   # (B,C) sources per clump
    # equilibrate, then average the agent-per-clump occupancy over the tail
    occ = torch.zeros(sim.B, C)
    nmeas = 0
    for t in range(T):
        sim.step()
        if t < T // 2:
            continue
        # IFD is about COMMITTED foragers: count only ACTIVE agents (seeds + recruited), the allocated
        # pool = the policy p_i. Inactive Levy explorers are the un-allocated 'free' agents and are
        # excluded (including them just adds a uniform dilution). Assign each active agent to its
        # nearest clump centre (min-image torus).
        active = sim._active_state                         # (B, N) bool
        d = sim._pair_dist(sim.pos, sim.clump_centers)     # (B, N, C)
        near = d.argmin(dim=2)                             # (B, N)
        o = torch.zeros(sim.B, C)
        o.scatter_add_(1, near, active.to(torch.float32))
        occ += o; nmeas += 1
    occ /= max(nmeas, 1)
    return rich.numpy(), occ.numpy()


def matching_slope(rich, occ):
    """Pool (clump, replica) with richness>0 and occupancy>0; regress log occ on log richness."""
    r = rich.flatten(); o = occ.flatten()
    m = (r >= 1) & (o >= 0.5)
    x, y = np.log(r[m]), np.log(o[m])
    if len(x) < 8:
        return np.nan, np.nan, 0
    A = np.polyfit(x, y, 1)
    yhat = np.polyval(A, x)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return A[0], r2, len(x)


def percapita_cv(rich, occ):
    """Coefficient of variation of per-capita share (occupancy/richness) across clumps.
    IFD equalises per-capita intake -> low CV; no matching -> high CV."""
    r = rich.flatten(); o = occ.flatten()
    m = (r >= 1) & (o >= 0.5)
    pc = o[m] / r[m]
    return pc.std() / (pc.mean() + 1e-9)


def main():
    print("Ideal Free Distribution test (CPU, N=200, C=10 clumps, Pareto richness, B=16, T=800)\n")
    for label, conv in [("recruiting (coupling ON) ", True), ("ablated  (coupling OFF)", False)]:
        rich, occ = run(conv)
        slope, r2, n = matching_slope(rich, occ)
        cv = percapita_cv(rich, occ)
        # richness spread actually realised
        rr = rich.flatten(); rr = rr[rr >= 1]
        print(f"{label}:")
        print(f"   richness spread: {rr.min():.0f}..{rr.max():.0f} sources/clump ({rr.max()/max(rr.min(),1):.0f}x)")
        print(f"   MATCHING slope (log agents vs log richness) = {slope:.2f}   R^2={r2:.2f}  (n={n})")
        print(f"     slope 1 = ideal free distribution; <1 undermatching; 0 = no matching")
        print(f"   per-capita-share CV across clumps = {cv:.2f}   (low = intake equalised = IFD)")
        print()
    print("Reading: recruiting slope near 1 with low CV = the swarm reaches the IFD / Nash rest point")
    print("of the habitat-selection game -- the replicator / policy-gradient fixed point. Ablation")
    print("collapsing the slope = the recruitment coupling is what performs the 'policy update'.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps", "ifd_test.npz")
    r_on, o_on = run(True); r_off, o_off = run(False)
    np.savez(out, rich_on=r_on, occ_on=o_on, rich_off=r_off, occ_off=o_off)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
