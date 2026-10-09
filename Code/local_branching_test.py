#!/usr/bin/env python3
"""
Direct test of LOCAL criticality (paper item 7 + 8): measure the per-cluster / per-avalanche
BRANCHING RATIO of the recruitment cascades, not just the cluster-size distribution.

Motivation. The paper argues "criticality is local": the swarm-averaged branching ratio is
subcritical (sigma ~ 0.08) yet the recruited clusters are scale-free. Scale-free sizes are only
*indirect* evidence. The direct test is the branching ratio of the cascades themselves: a critical
branching process has sigma = 1 and produces power-law avalanche sizes with exponent tau ~ 3/2.

Estimators (all on the SAME running foraging model that produces the Fig-3 clusters):
  * sigma_avalanche  -- Galton-Watson MLE per connected component (avalanche):
        sigma = sum_c (size_c - seeds_c) / sum_c (size_c - lastgen_c)
    over components that actually branched (size >= 2, >=1 recruitment round). = offspring/parents;
    -> 1 at criticality. Excludes size-1 seed events, which carry no branching information.
  * sigma_swarm      -- the model's own per-tick sigma (branch_num/branch_den = recruited per
        active-agent-round, summed over all agents incl. non-branching seeds/blocked). This is the
        ~0.08 "global" number: the SAME criticality diluted by the non-branching majority.
  * local mean degree inside cascaded clusters (item 8) -- compare to the blocking boundary 2/theta.
  * cluster-size exponent tau (should be ~3/2 = mean-field critical branching).

Generation sizes come from mobile_cascade's optional, behaviour-neutral `_record_generations` hook.
CPU only (no GPU lock needed). Usage: python3 local_branching_test.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobile_cascade import MobileCascade, MobileCascadeConfig  # noqa: E402


@torch.no_grad()
def components(adj, active):
    """Connected-component labels of the active subgraph. Returns (B,N) long labels (or -1 inactive)."""
    B, N, _ = adj.shape
    dev = adj.device
    conn = (adj & active.unsqueeze(1) & active.unsqueeze(2))
    conn = conn | torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
    labels = torch.arange(N, device=dev).float().unsqueeze(0).expand(B, N).clone()
    INF = float(N + 1)
    for _ in range(N):
        lab_j = labels.unsqueeze(1).expand(B, N, N)
        new = torch.where(conn, lab_j, torch.full((), INF, device=dev)).min(dim=2).values
        if torch.equal(new, labels):
            break
        labels = new
    lab = labels.long()
    lab[~active] = -1
    return lab


@torch.no_grad()
def tick_stats(lab, gidx, deg, N, big):
    """Vectorized per-tick cascade statistics (replaces the per-component Python loop).
    Returns (sizes, offspring, parents, local_deg, replica_idx) for the cascaded components this tick:
    components with size>=2, a recruitment round (lastgen>=1), and parents>0 -- identical criteria to the
    original loop. Uses global component ids cg=b*N+lab so all replicas are handled at once via bincount."""
    B = lab.shape[0]
    dev = lab.device
    am = lab >= 0                                                # active nodes
    b_idx = torch.arange(B, device=dev).view(B, 1).expand_as(lab)
    cg = (b_idx * N + lab)[am]                                   # (M,) global component id per active node
    gi = gidx[am]                                               # (M,) generation index (may be -1)
    dg = deg[am].to(torch.float32)
    BN = B * N
    size = torch.bincount(cg, minlength=BN)
    seeds = torch.bincount(cg[gi == 0], minlength=BN)
    lastgen = torch.full((BN,), -1, device=dev, dtype=gi.dtype)
    lastgen = lastgen.scatter_reduce(0, cg, gi, reduce="amax", include_self=True)
    is_last = gi == lastgen[cg]                                 # node is in its component's final generation
    lastgen_c = torch.bincount(cg[is_last], minlength=BN)
    degsum = torch.bincount(cg, weights=dg, minlength=BN)
    keep = (size >= 2) & (lastgen >= 1) & ((size - lastgen_c) > 0)
    comp = keep.nonzero(as_tuple=False).squeeze(1)
    sizes = size[comp].to(torch.float32)
    offspring = (size[comp] - seeds[comp]).to(torch.float32)
    parents = (size[comp] - lastgen_c[comp]).to(torch.float32)
    local_deg = degsum[comp] / sizes
    rep = (comp // N)
    return sizes, offspring, parents, local_deg, rep


@torch.no_grad()
def run_setting(theta, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, phi=args.phi,
                              theta=theta, step=args.step, pull=args.pull,
                              consumption=args.consumption, capacity=args.capacity,
                              converge=True, seed=args.seed, device=args.device,
                              levy_alpha=args.levy, exploit_step=args.exploit_step,
                              clump_count=args.clump_count, clump_radius=args.clump_radius,
                              clump_beta=args.clump_beta)
    sim = MobileCascade(cfg)
    sim._record_generations = True
    t0 = int((1.0 - args.tail) * args.T)

    # vectorized accumulators: append per-tick device tensors, concatenate once at the end
    S, O, P, Dg, R = [], [], [], [], []
    sigma_swarm = []     # model's own per-tick sigma (the ~0.08 estimator), tick means
    for t in range(args.T):
        sim.step()
        if t < t0:
            continue
        gens = sim._generations                     # list of (B,N) bool masks; gens[0]=seeds
        adj = sim._adj
        active = sim._active_state
        deg = adj.sum(dim=2)                          # (B,N)
        lab = components(adj, active)                 # (B,N)
        gidx = torch.full_like(lab, -1)               # generation of each active node (gidx[b,i]=r)
        for r, g in enumerate(gens):
            gidx[g] = r
        sigma_swarm.append(float(sim._sigma.mean()))
        s, o, p, d, rp = tick_stats(lab, gidx, deg, args.N, args.big)
        if s.numel():
            S.append(s); O.append(o); P.append(p); Dg.append(d); R.append(rp)

    if S:
        sizes = torch.cat(S).cpu().numpy().astype(float)
        off_np = torch.cat(O).cpu().numpy().astype(float)
        par_np = torch.cat(P).cpu().numpy().astype(float)
        rep_np = torch.cat(R).cpu().numpy().astype(int)
        off_sum = float(off_np.sum()); par_sum = max(float(par_np.sum()), 1e-9)
        rep_off = np.bincount(rep_np, weights=off_np, minlength=args.B)
        rep_par = np.bincount(rep_np, weights=par_np, minlength=args.B)
        big_mask = sizes >= args.big
        per_aval_sigma = off_np[big_mask] / np.maximum(par_np[big_mask], 1e-9)
        local_deg = torch.cat(Dg).cpu().numpy().tolist()
    else:
        sizes = np.array([]); off_sum, par_sum = 0.0, 1e-9
        rep_off = np.zeros(args.B); rep_par = np.zeros(args.B)
        per_aval_sigma = np.array([]); local_deg = []
    all_sizes = sizes

    off_sum = max(off_sum, 0.0); par_sum = max(par_sum, 1e-9)
    sigma_aval = off_sum / par_sum
    # bootstrap CI over the B INDEPENDENT replicas (block-bootstrap; ticks within a replica are
    # autocorrelated, replicas are not)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(4000):
        idx = rng.integers(0, args.B, args.B)
        p = rep_par[idx].sum()
        boots.append(rep_off[idx].sum() / p if p > 0 else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    pa = np.array(per_aval_sigma, dtype=float)
    pa_med = float(np.median(pa)) if len(pa) else np.nan
    pa_iqr = (float(np.percentile(pa, 25)), float(np.percentile(pa, 75))) if len(pa) else (np.nan, np.nan)

    sizes = np.array(all_sizes, dtype=float)
    # tau from the cluster-size PDF tail (MLE-ish via CCDF slope), sizes>=2
    tau = np.nan
    if len(sizes) > 50:
        x = np.sort(sizes)
        c = 1.0 - np.arange(len(x)) / len(x)
        m = (x >= 2) & (x <= x.max() * 0.5) & (c > 0)
        if m.sum() > 5:
            b = np.polyfit(np.log10(x[m]), np.log10(c[m]), 1)
            tau = 1.0 - b[0]                          # PDF exponent = 1 - CCDF slope

    return {
        "theta": theta,
        "sigma_aval": sigma_aval, "sigma_aval_lo": lo, "sigma_aval_hi": hi,
        "sigma_aval_med": pa_med, "sigma_aval_q25": pa_iqr[0], "sigma_aval_q75": pa_iqr[1],
        "sigma_swarm": float(np.mean(sigma_swarm)),
        "per_aval_sigma": np.array(per_aval_sigma, dtype=np.float32),
        "local_deg_mean": float(np.mean(local_deg)) if local_deg else np.nan,
        "block_boundary_2_over_theta": 2.0 / theta,
        "tau": tau, "n_cascaded_clusters": len(sizes),
        "n_big_avalanches": len(pa),
        "sizes": sizes.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=150)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=700)
    ap.add_argument("--tail", type=float, default=0.5)
    ap.add_argument("--rho_r", type=float, default=3.0)
    ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--clump-count", type=int, default=None, dest="clump_count",
                    help="sources in this many spatial clumps (patchy field); None = uniform")
    ap.add_argument("--clump-radius", type=float, default=40.0, dest="clump_radius")
    ap.add_argument("--clump-beta", type=float, default=None, dest="clump_beta",
                    help="if set, scale-free clumped field (Pareto clump weights, exponent ~beta)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.20, 0.35])
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1)
    ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--capacity", type=float, default=40.0)
    ap.add_argument("--levy", type=float, default=None, help="Levy exponent; None = random walk")
    ap.add_argument("--exploit-step", type=float, default=3.0, dest="exploit_step")
    ap.add_argument("--big", type=int, default=8, help="min size for the per-avalanche sigma distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="local_branching")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    mode = "Levy(a=%.2f)" % args.levy if args.levy is not None else "random walk"
    print(f"local branching: N={args.N} K={args.K} rho_r={args.rho_r} phi={args.phi} "
          f"thetas={args.thetas}  exploration={mode}", flush=True)
    save = {"thetas": np.array(args.thetas), "N": np.array([args.N]),
            "levy": np.array([args.levy if args.levy is not None else -1.0])}
    for th in args.thetas:
        r = run_setting(th, args)
        for k in ["sigma_aval", "sigma_aval_lo", "sigma_aval_hi", "sigma_aval_med",
                  "sigma_aval_q25", "sigma_aval_q75", "sigma_swarm", "local_deg_mean",
                  "block_boundary_2_over_theta", "tau", "n_cascaded_clusters", "n_big_avalanches"]:
            save[f"{k}_{th:.2f}"] = np.array([r[k]])
        save[f"per_aval_sigma_{th:.2f}"] = r["per_aval_sigma"]
        save[f"sizes_{th:.2f}"] = r["sizes"]
        print(f"  theta={th:.2f}: sigma_AVALANCHE={r['sigma_aval']:.3f} "
              f"[{r['sigma_aval_lo']:.3f},{r['sigma_aval_hi']:.3f}] (replica-boot)  "
              f"per-avalanche median={r['sigma_aval_med']:.2f} "
              f"IQR[{r['sigma_aval_q25']:.2f},{r['sigma_aval_q75']:.2f}] (n={r['n_big_avalanches']})  "
              f"|  sigma_swarm(global)={r['sigma_swarm']:.3f}  "
              f"local_deg={r['local_deg_mean']:.2f} (2/theta={2.0/th:.2f})  "
              f"tau={r['tau']:.2f}  n_clusters={r['n_cascaded_clusters']}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez(path, **save)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
