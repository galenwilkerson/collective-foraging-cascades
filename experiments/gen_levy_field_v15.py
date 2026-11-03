#!/usr/bin/env python3
"""
RQ4 Q4c (SI levy_field) on the CORRECTED engine -> v15. Question: is heavy-tailed search advantageous on
CLUSTERED fields, and what alpha maximizes harvest per field?

Sweep the Levy stable index alpha; measure steady-state harvest (FIXED-HORIZON tail window) for:
  CLUSTERED  = few tight clumps (clump_count=3, cover<1)
  SCATTERED  = CSR / Poisson (clump_count=None) -- the regime where inverse-square Levy is provably NOT
               optimal (Levernier 2020); the honest contrast, not merely "more clumps".
Recruitment ON (converge=True): this is the swarm-level alpha*(field). Canonical operating point
(rho_r=3, theta=0.20, phi=0.05). alpha grid extended DOWN to 0.3 because a strongly-clustered optimum may
be ballistic (alpha*<0.6) and would otherwise pin at the old 0.6 floor. GPU, per-replica harvest for CIs.
Establishes alpha*(field) from scratch; reports whatever direction it shows.
"""
import os, sys, time, argparse
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, _P.CODE)
from mobile_cascade import MobileCascade, MobileCascadeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "sweeps"); FIGDIR = os.path.join(os.path.dirname(HERE), "figures")


@torch.no_grad()
def harvest(alpha, clump_count, args):
    cfg = MobileCascadeConfig(B=args.B, N=args.N, K=args.K, L=args.L, rho_r=args.rho_r, phi=args.phi,
        theta=args.theta, step=args.step, pull=args.pull, consumption=args.consumption,
        capacity=args.capacity, converge=True, seed=args.seed, device=args.device, levy_alpha=alpha,
        exploit_step=args.exploit_step, clump_count=clump_count, clump_radius=args.clump_radius)
    o = MobileCascade(cfg).run(args.T)
    t0 = int((1.0 - args.tail) * args.T); Tw = args.T - t0
    return o["consumed"][:, t0:].sum(1).cpu().numpy() / (args.consumption * args.N * Tw)   # (B,) per-rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=150); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--B", type=int, default=48); ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--tail", type=float, default=0.5); ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--rho_r", type=float, default=3.0); ap.add_argument("--phi", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=0.20); ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--pull", type=float, default=0.1); ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--capacity", type=float, default=40.0); ap.add_argument("--exploit_step", type=float, default=3.0)
    ap.add_argument("--clump-radius", type=float, default=25.0, dest="clump_radius")
    ap.add_argument("--clustered-clumps", type=int, default=3, dest="clustered")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.15, 0.25, 0.4, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cover_cl = args.clustered * np.pi * (args.clump_radius / args.L) ** 2
    print(f"levy_field: clustered={args.clustered} clumps (cover={cover_cl:.3f}<1) vs scattered=CSR; "
          f"alphas={args.alphas}", flush=True)
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); res = {}
    for name, cc in [("clustered", args.clustered), ("scattered (CSR)", None)]:
        reps = np.array([harvest(a, cc, args) for a in args.alphas])     # (n_alpha, B)
        m = reps.mean(1); astar = args.alphas[int(np.argmax(m))]
        res[name] = (m, reps.std(1) / np.sqrt(reps.shape[1]))
        print(f"  {name:16s}: harvest={np.array2string(m, precision=3)}  -> alpha*={astar}", flush=True)
    os.makedirs(DATADIR, exist_ok=True)
    np.savez(os.path.join(DATADIR, "levy_field_v15.npz"), alphas=np.array(args.alphas),
             **{k.split()[0]: v[0] for k, v in res.items()})
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for (name, (m, sem)), col in zip(res.items(), ["#c0392b", "#1f4e79"]):
        ax.errorbar(args.alphas, m, yerr=sem, marker="o", lw=1.8, color=col, capsize=2,
                    label=f"{name} ($\\alpha^*$={args.alphas[int(np.argmax(m))]})")
    ax.set_xlabel("Lévy stable index $\\alpha$ (small = heavier tail)"); ax.set_ylabel("steady-state harvest")
    ax.set_title("Heavy-tailed search vs field clustering"); ax.legend()
    fig.tight_layout(); os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_levy_field.pdf"); fig.savefig(out)
    vram = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0
    print(f"wrote {out}  ({time.time()-t0:.0f}s, VRAM {vram:.2f}GB)")


if __name__ == "__main__":
    main()
