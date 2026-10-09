#!/usr/bin/env python3
"""B.1 -- "the signal outruns the bodies": recruitment-front speed vs agent-locomotion speed.

Instruments the near-critical foraging model, logging per-tick positions (B,N,2 on an L-torus) and the
active mask (B,N). Then, after burn-in, measures:

  v_agent  -- per-tick agent displacement |min_image(pos_t - pos_{t-1})| (all agents; and the Levy-tail
              fast movers via high percentiles) = how far a BODY moves in one tick.
  reach    -- for each agent NEWLY active at t (active_t & ~active_{t-1}), the min-image distance to the
              nearest agent that was active at t-1 (its recruiter) = how far the RECRUITMENT SIGNAL jumped
              across space in one tick (one graph hop, up to the interaction radius r).
  R_active -- radius (max member-to-centroid min-image distance) of the largest active connected cluster
              each tick = the spatial extent coordination reaches.

Claim: recruitment reach per tick >> agent displacement per tick (signal outruns bodies), and active
clusters span distances an agent could not traverse in the cascade's lifetime. CPU. Data -> /mnt/newssd.
"""
import os, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402
from local_branching_test import components                     # reuse tested component labeller

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")


def minimg_norm(d, L):
    d = (d + L / 2.0) % L - L / 2.0
    return np.linalg.norm(d, axis=-1)


@torch.no_grad()
def run(N=100, B=16, T=500, tail=0.4, theta=0.20, levy=1.5, rho_r=3.0, phi=0.05,
        L=800.0, step=20.0, exploit_step=3.0, seed=0):
    cfg = MobileCascadeConfig(B=B, N=N, K=max(2, N // 5), L=L, rho_r=rho_r, phi=phi, theta=theta,
                              step=step, pull=0.1, consumption=1.0, capacity=40.0, converge=True,
                              seed=seed, device="cpu", levy_alpha=levy, exploit_step=exploit_step)
    sim = MobileCascade(cfg)
    t0 = int((1.0 - tail) * T)
    POS, ACT = [], []
    for t in range(T):
        sim.step()
        if t >= t0:
            POS.append(sim.pos.cpu().numpy().copy())            # (B,N,2)
            ACT.append(sim._active_state.cpu().numpy().copy())  # (B,N) bool
    return np.array(POS), np.array(ACT), float(sim.r), L, step, exploit_step


def analyze(POS, ACT, r, L):
    Tn, B, N, _ = POS.shape
    v_agent, reach, Ractive = [], [], []
    for t in range(1, Tn):
        disp = minimg_norm(POS[t] - POS[t - 1], L)              # (B,N) per-agent displacement this tick
        v_agent.append(disp.ravel())
        for b in range(B):
            now, prev = ACT[t, b], ACT[t - 1, b]
            newly = now & (~prev)
            src = prev                                          # agents active last tick = recruiters
            if newly.any() and src.any():
                pn = POS[t, b][newly]                           # newly-active positions (use current pos)
                ps = POS[t - 1, b][src]                         # recruiter positions (previous tick)
                d = pn[:, None, :] - ps[None, :, :]             # (n_new, n_src, 2)
                dd = minimg_norm(d, L)                          # (n_new, n_src)
                reach.append(dd.min(axis=1))                    # nearest recruiter distance
        # largest active cluster radius (reuse the tested component labeller on this tick)
        adj = torch.from_numpy((minimg_norm(POS[t][:, :, None, :] - POS[t][:, None, :, :], L) <= r))
        active_t = torch.from_numpy(ACT[t])
        lab = components(adj, active_t)                          # (B,N)
        for b in range(B):
            labs = lab[b]
            for cid in torch.unique(labs[labs >= 0]).tolist():
                mask = (labs == cid).numpy()
                if mask.sum() < 3:
                    continue
                pts = POS[t, b][mask]
                c = pts.mean(0)
                Ractive.append(float(minimg_norm(pts - c, L).max()))
    v_agent = np.concatenate(v_agent)
    reach = np.concatenate(reach) if reach else np.array([np.nan])
    Ractive = np.array(Ractive) if Ractive else np.array([np.nan])
    return v_agent, reach, Ractive


def main():
    POS, ACT, r, L, step, exploit = run()
    v_agent, reach, Ractive = analyze(POS, ACT, r, L)
    q = lambda x, p: float(np.nanpercentile(x, p))
    print(f"interaction radius r = {r:.1f} (world units); L={L:.0f}; Levy step scale={step}, exploit step={exploit}")
    print(f"agent displacement / tick  v_agent : median={q(v_agent,50):.2f}  "
          f"90th={q(v_agent,90):.2f}  99th={q(v_agent,99):.2f}  mean={np.nanmean(v_agent):.2f}")
    print(f"recruitment reach  / tick  (signal): median={q(reach,50):.2f}  "
          f"90th={q(reach,90):.2f}  99th={q(reach,99):.2f}  mean={np.nanmean(reach):.2f}")
    print(f"active-cluster radius R_active      : median={q(Ractive,50):.2f}  90th={q(Ractive,90):.2f}  max={np.nanmax(Ractive):.2f}")
    print(f"--> signal/body speed ratio (median reach / median v_agent) = {q(reach,50)/max(q(v_agent,50),1e-9):.2f}x")
    print(f"--> cluster radius / per-tick body displacement (median)    = {q(Ractive,50)/max(q(v_agent,50),1e-9):.2f}x")
    np.savez(os.path.join(DATA, "b1_frontspeed_raw.npz"),
             v_agent=v_agent.astype(np.float32), reach=reach.astype(np.float32),
             Ractive=Ractive.astype(np.float32), r=r, L=L, step=step, exploit=exploit)
    print("wrote", os.path.join(DATA, "b1_frontspeed_raw.npz"))


if __name__ == "__main__":
    main()
