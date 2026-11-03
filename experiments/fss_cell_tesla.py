#!/usr/bin/env python3
"""Tesla-safe copy of fss_cell.py: identical logic, but vram_B targets ~16 GB (not the A100's 30 GB) so
large-N cells fit the shared RTX 3090 (24 GB) with headroom for other projects. For N<=400 B still caps
at 1024 (same as the A100 run); only N=800 (B~567) and N=3200 (B~40) get a smaller batch. Resumable
checkpoints mean a smaller B is compensated by running more chunks. Code/ must be on PYTHONPATH."""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_branching_test import run_setting


def vram_B(N):
    # ~44 bytes/(B*N^2) worst case (measured 2026-11-01). Target ~16 GB of the 24 GB RTX 3090.
    return int(np.clip(16e9 / (44.1 * N * N), 16, 1024))


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
    with open(tmp, "wb") as fh:
        np.savez(fh, N=a.N, theta=a.theta, B=vram_B(a.N), done_chunks=nc, chunks=a.chunks,
                 sigma_aval=off_sum/max(par_sum, 1e-9), off_sum=off_sum, par_sum=par_sum,
                 sizes=allsz.astype(np.float32), swarm=np.array(swarm), locdeg=np.array(locdeg), n_sizes=len(allsz))
    os.replace(tmp, ckpt)


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
          f"n_sizes={sum(len(s) for s in sizes_parts)} B={vram_B(a.N)}", flush=True)
print(f"DONE cell N={a.N} th={a.theta:.3f}: sigma_aval={off_sum/max(par_sum,1e-9):.3f} -> {ckpt}", flush=True)
