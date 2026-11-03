#!/usr/bin/env python3
"""
One (N, theta) cell of the tuned cascade-size FSS collapse, CHUNKED + CHECKPOINTED, on the CORRECTED engine.
Runs `chunks` short run_setting sub-runs (fresh seeds), accumulating cascade sizes + raw off/par sums (for an
exact pooled sigma_aval), writing an ATOMIC checkpoint after every chunk. On Slurm --requeue (after a modest
--time kill) it RESUMES from the last completed chunk, so no finished work is lost and short --time backfills.
Imports the staged corrected Code/mobile_cascade via local_branching_test (short-step, near-CSR, alpha=1.5).
NOTATION: theta = activation threshold (paper phi); config phi = coverage (0.05).
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # staged: mobile_cascade, batched_swarm, local_branching_test
from local_branching_test import run_setting

def vram_B(N):
    # Peak ~44 bytes/(B*N^2) at the WORST-case (most-active, low-theta) cell: the engine holds several
    # (B,N,N[,2]) tensors at once -- pairwise-distance diff, adjacency, min-image temp, beacon x beacon
    # (grows with the active count). Target ~30GB of the A100-40GB, leaving headroom for context/frag.
    # Measured 2026-11-01 on Tesla (N=3200, B=32, theta=0.08 -> 14.46 GB peak => 44.1 B/(B*N^2)).
    return int(np.clip(30e9 / (44.1 * N * N), 16, 1024))

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--theta", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunks", type=int, default=6)
ap.add_argument("--tchunk", type=int, default=120)
ap.add_argument("--seed0", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ckpt = os.path.join(a.out, f"cell_N{a.N}_th{a.theta:.3f}.npz")

done, sizes_parts, off_sum, par_sum, swarm, locdeg = 0, [], 0.0, 0.0, [], []
if os.path.exists(ckpt):
    d = np.load(ckpt, allow_pickle=True)
    done = int(d["done_chunks"])
    if len(d["sizes"]): sizes_parts = [d["sizes"]]
    off_sum, par_sum = float(d["off_sum"]), float(d["par_sum"])
    swarm, locdeg = list(d["swarm"]), list(d["locdeg"])
    print(f"RESUME N={a.N} th={a.theta:.3f} from chunk {done}/{a.chunks}", flush=True)
    if done >= a.chunks:
        print(f"already complete: sigma_aval={off_sum/max(par_sum,1e-9):.3f}", flush=True); sys.exit(0)

def save(nc):
    allsz = np.concatenate([s for s in sizes_parts if len(s)]) if any(len(s) for s in sizes_parts) else np.array([], np.float32)
    tmp = ckpt + ".tmp"
    with open(tmp, "wb") as fh:   # file handle => np.savez writes EXACTLY to tmp (no auto ".npz" suffix)
        np.savez(fh, N=a.N, theta=a.theta, B=vram_B(a.N), done_chunks=nc, chunks=a.chunks,
                 sigma_aval=off_sum/max(par_sum, 1e-9), off_sum=off_sum, par_sum=par_sum,
                 sizes=allsz.astype(np.float32), swarm=np.array(swarm), locdeg=np.array(locdeg), n_sizes=len(allsz))
    os.replace(tmp, ckpt)   # atomic: a kill mid-write cannot corrupt the checkpoint

for c in range(done, a.chunks):
    args = argparse.Namespace(N=a.N, K=30, B=vram_B(a.N), T=a.tchunk, tail=0.5, rho_r=3.0, phi=0.05, L=800.0,
        device="cuda", clump_count=None, clump_radius=40.0, clump_beta=None, step=20.0, pull=0.1,
        consumption=1.0, capacity=40.0, levy=1.5, exploit_step=3.0, big=8, seed=a.seed0 + c)
    r = run_setting(a.theta, args)
    sizes_parts.append(np.asarray(r["sizes"], dtype=np.float32))
    off_sum += float(r["off_sum"]); par_sum += float(r["par_sum"]); swarm.append(float(r["sigma_swarm"]))
    if r["local_deg_mean"] == r["local_deg_mean"]: locdeg.append(float(r["local_deg_mean"]))
    save(c + 1)
    print(f"  chunk {c+1}/{a.chunks} N={a.N} th={a.theta:.3f}: cum sigma_aval={off_sum/max(par_sum,1e-9):.3f} "
          f"n_sizes={sum(len(s) for s in sizes_parts)}", flush=True)
print(f"DONE cell N={a.N} th={a.theta:.3f}: sigma_aval={off_sum/max(par_sum,1e-9):.3f} -> {ckpt}", flush=True)
