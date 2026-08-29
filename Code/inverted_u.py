#!/usr/bin/env python3
"""
Q1 validation: reproduce the LTM cascade window (inverted-U) from the model's own
primitives. For each mean degree z (set via the connectivity density), build a random
geometric graph on the torus, seed a small perturbation, run the linear-threshold cascade
to its fixed point, and measure:

    cascade fraction   = fraction of nodes activated (mean +/- std over replicas)
    GCC fraction       = size of the largest connected component / N

The prediction (Watts' cascade window, here on an RGG): cascade fraction rises past the
percolation threshold z_c^RGG ~ 4.51 and falls at high z (blocking, k > 1/theta), peaking at
intermediate connectivity. This should reproduce `figure2_spatial_cascades_combined`.

Primitives (minimum-image distance, symmetric RGG edges, LTM fixed point) mirror
`batched_swarm.py`. The batched connected-components (label propagation) is reused later for
the largest-active-cluster order parameter Psi.

Usage:
    python3 inverted_u.py                 # default sweep -> sweeps/inverted_u.csv + .png
    python3 inverted_u.py --quick
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_swarm import select_device  # noqa: E402

Z_C_RGG = 4.51  # continuum-percolation critical mean degree for a 2D RGG


def min_image_dist(pos, size):
    """Toroidal pairwise distances for pos (B,N,2) -> (B,N,N)."""
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)          # (B,N,N,2)
    diff = diff - size * torch.round(diff / size)
    return diff.norm(dim=-1)


def build_rgg(pos, r, p, size, gen=None):
    """Undirected RGG adjacency (B,N,N) bool: within radius r AND symmetric edge prob p."""
    B, N, _ = pos.shape
    within = min_image_dist(pos, size) <= r
    eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
    if p >= 1.0:
        return within & (~eye)
    rp = torch.rand(B, N, N, device=pos.device, generator=gen)
    rs = torch.triu(rp, diagonal=1)
    rs = rs + rs.transpose(1, 2)
    return within & (rs < p) & (~eye)


def ltm_fixed_point(adj, seeds, theta):
    """Run the LTM to its fixed point. adj (B,N,N) bool, seeds (B,N) bool. Returns active (B,N)."""
    B, N, _ = adj.shape
    degree = adj.sum(dim=2).to(torch.float32)
    inv_deg = torch.where(degree > 0, 1.0 / degree.clamp(min=1.0), torch.zeros_like(degree))
    active = seeds.clone()
    for _ in range(N):
        a = (adj & active.unsqueeze(1)).sum(dim=2).to(torch.float32)
        newly = (a * inv_deg >= theta) & (~active)
        if not bool(newly.any()):
            break
        active = active | newly
    return active


def gcc_fraction(adj):
    """Largest connected-component fraction per replica, via label propagation. adj (B,N,N)."""
    B, N, _ = adj.shape
    labels = torch.arange(N, device=adj.device).unsqueeze(0).expand(B, N).to(torch.float32).clone()
    eye = torch.eye(N, dtype=torch.bool, device=adj.device).unsqueeze(0)
    conn = adj | eye
    INF = float(N + 1)
    for _ in range(N):
        lab_j = labels.unsqueeze(1).expand(B, N, N)
        new_labels = torch.where(conn, lab_j, torch.full((), INF, device=adj.device)).min(dim=2).values
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    gcc = torch.empty(B, device=adj.device)
    lab = labels.long()
    for b in range(B):
        gcc[b] = torch.bincount(lab[b], minlength=N).max().float() / N
    return gcc


@torch.no_grad()
def sweep(z_grid, *, N, B, theta, p, seed_frac, L=1.0, device=None, seed=0):
    dev = select_device(device)
    gen = torch.Generator(device=dev) if dev.type in ("cpu", "cuda") else None
    if gen is not None:
        gen.manual_seed(seed)
    size = torch.tensor([L, L], device=dev)
    rows = []
    n_seed = max(1, int(round(seed_frac * N)))
    for z in z_grid:
        # z = pi * p * rho_r, rho_r = N (r/L)^2  ->  r = L sqrt(rho_r / N)
        rho_r = z / (math.pi * p)
        r = L * math.sqrt(rho_r / N)
        pos = (torch.rand(B, N, 2, device=dev, generator=gen) * size)
        adj = build_rgg(pos, r, p, size, gen)
        z_obs = adj.sum(dim=2).to(torch.float32).mean().item()
        # seed n_seed random nodes per replica
        seeds = torch.zeros(B, N, dtype=torch.bool, device=dev)
        idx = torch.argsort(torch.rand(B, N, device=dev, generator=gen), dim=1)[:, :n_seed]
        seeds.scatter_(1, idx, True)
        active = ltm_fixed_point(adj, seeds, theta)
        casc = active.sum(dim=1).to(torch.float32) / N            # (B,)
        gcc = gcc_fraction(adj)                                    # (B,)
        rows.append((z, z_obs, casc.mean().item(), casc.std().item(),
                     gcc.mean().item(), gcc.std().item()))
        print(f"z={z:5.2f} z_obs={z_obs:6.2f} cascade={casc.mean():.3f}+/-{casc.std():.3f} "
              f"gcc={gcc.mean():.3f}", flush=True)
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zmin", type=float, default=0.5)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--theta", type=float, default=0.18)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--seed-frac", type=float, default=0.01, help="fraction of nodes seeded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="inverted_u")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.nz, args.N, args.B = 12, 200, 32

    z_grid = np.linspace(args.zmin, args.zmax, args.nz).tolist()
    print(f"inverted-U sweep: {args.nz} z x N={args.N} x B={args.B}, theta={args.theta}, "
          f"seed_frac={args.seed_frac}", flush=True)
    res = sweep(z_grid, N=args.N, B=args.B, theta=args.theta, p=args.p,
                seed_frac=args.seed_frac, device=args.device, seed=args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, f"{args.tag}.csv")
    with open(csv, "w") as fh:
        fh.write("z_set,z_obs,cascade_mean,cascade_std,gcc_mean,gcc_std\n")
        for row in res:
            fh.write(",".join(f"{x:.6g}" for x in row) + "\n")
    print(f"wrote {csv}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z_obs, cm, cs, gm = res[:, 1], res[:, 2], res[:, 3], res[:, 4]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z_obs, cm, "-", color="C0", label="mean cascade fraction")
        ax.fill_between(z_obs, cm - cs, cm + cs, color="C0", alpha=0.2, label=r"$\pm1$ std")
        ax.axvline(Z_C_RGG, ls=":", color="k", lw=1)
        ax.text(Z_C_RGG, 0.02, r"$z_c^{\rm RGG}$", rotation=0, va="bottom")
        ax2 = ax.twinx()
        ax2.plot(z_obs, gm, "--", color="C0", label="GCC fraction")
        ax2.set_ylabel("Giant component fraction")
        ax.set_xlabel("Mean degree (observed)")
        ax.set_ylabel("Cascade fraction")
        ax.set_title("LTM cascade window on an RGG (Q1)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        png = os.path.join(args.outdir, f"{args.tag}.png")
        fig.savefig(png, dpi=130)
        print(f"wrote {png}", flush=True)
    except ImportError:
        print("matplotlib unavailable; skipped plot", flush=True)


if __name__ == "__main__":
    main()
