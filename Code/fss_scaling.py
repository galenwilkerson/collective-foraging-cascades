#!/usr/bin/env python3
"""
Finite-size scaling / universality of the cascade window.

The paper's claims are stated in dimensionless controls (mean degree z, threshold theta, ...).
This checks that they are genuinely N-independent -- i.e. properties of a class of systems, not
of one system size. For several agent counts N we sweep the mean degree z and measure, over an
ensemble of random geometric graphs + LTM fixed points:

    cascade fraction   -- should COLLAPSE onto one master curve across N (window is dimensionless)
    GCC fraction       -- percolation onset; curves for different N CROSS near z_c^RGG ~ 4.51
    Psi = largest active-cluster fraction  -- order parameter
    chi = N * Var(Psi) -- susceptibility; a PEAK that GROWS with N locates a (finite-size) critical
                          connectivity -> the standard FSS signature.

Reuses the primitives in inverted_u.py. Saves sweeps/fss_scaling.npz + .png.

Usage: python3 fss_scaling.py
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inverted_u import build_rgg, ltm_fixed_point, select_device  # noqa: E402


@torch.no_grad()
def psi_largest_active(adj, active):
    """Largest connected component of the ACTIVE subgraph, as a fraction of N, per replica."""
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
    psi = torch.zeros(B, device=dev)
    for b in range(B):
        act = active[b]
        if int(act.sum()) == 0:
            continue
        psi[b] = torch.bincount(lab[b][act], minlength=N).max().float() / N
    return psi


@torch.no_grad()
def sweep_N(N, z_grid, *, B, theta, p, n_seed, L=1.0, device=None, seed=0):
    dev = select_device(device)
    gen = torch.Generator(device=dev) if dev.type in ("cpu", "cuda") else None
    if gen is not None:
        gen.manual_seed(seed)
    size = torch.tensor([L, L], device=dev)
    # single seed (fixed across N) gives the true Watts cascade window incl. the blocking upper bound
    casc, gcc_frac, chi, psi_mean = [], [], [], []
    for z in z_grid:
        rho_r = z / (math.pi * p)
        r = L * math.sqrt(rho_r / N)
        pos = torch.rand(B, N, 2, device=dev, generator=gen) * size
        adj = build_rgg(pos, r, p, size, gen)
        seeds = torch.zeros(B, N, dtype=torch.bool, device=dev)
        idx = torch.argsort(torch.rand(B, N, device=dev, generator=gen), dim=1)[:, :n_seed]
        seeds.scatter_(1, idx, True)
        active = ltm_fixed_point(adj, seeds, theta)
        cf = active.sum(dim=1).to(torch.float32) / N
        psi = psi_largest_active(adj, active)
        # GCC of the full graph (percolation), reuse Psi machinery on all-active
        allon = torch.ones(B, N, dtype=torch.bool, device=dev)
        gcc = psi_largest_active(adj, allon)
        casc.append(cf.mean().item())
        gcc_frac.append(gcc.mean().item())
        psi_mean.append(psi.mean().item())
        chi.append((N * psi.var(unbiased=True)).item())
    return (np.array(casc), np.array(gcc_frac), np.array(psi_mean), np.array(chi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ns", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--zmin", type=float, default=1.0)
    ap.add_argument("--zmax", type=float, default=14.0)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--B", type=int, default=384)
    ap.add_argument("--theta", type=float, default=0.18)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--n-seed", type=int, default=1, dest="n_seed", help="seeds per replica (fixed across N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps"))
    ap.add_argument("--tag", type=str, default="fss_scaling")
    args = ap.parse_args()

    z_grid = np.linspace(args.zmin, args.zmax, args.nz)
    print(f"FSS: Ns={args.Ns}  z in [{args.zmin},{args.zmax}] x{args.nz}  B={args.B}  theta={args.theta}",
          flush=True)
    out = {"z": z_grid, "Ns": np.array(args.Ns), "theta": np.array([args.theta])}
    for N in args.Ns:
        casc, gcc, psi, chi = sweep_N(N, z_grid, B=args.B, theta=args.theta, p=args.p,
                                      n_seed=args.n_seed, seed=args.seed)
        out[f"cascade_{N}"] = casc
        out[f"gcc_{N}"] = gcc
        out[f"psi_{N}"] = psi
        out[f"chi_{N}"] = chi
        zc = z_grid[int(np.argmax(chi))]
        print(f"  N={N:4d}: cascade peak={casc.max():.3f}  chi peak={chi.max():.3g} at z={zc:.2f}",
              flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, f"{args.tag}.npz"), **out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.3))
        for N in args.Ns:
            a1.plot(z_grid, out[f"cascade_{N}"], "-o", ms=3, label=f"N={N}")
            a2.plot(z_grid, out[f"gcc_{N}"], "-o", ms=3, label=f"N={N}")
            a3.plot(z_grid, out[f"chi_{N}"], "-o", ms=3, label=f"N={N}")
        a1.axvline(4.51, ls=":", color="k"); a1.set_title("cascade fraction (collapses across N)")
        a1.set_xlabel("mean degree z"); a1.set_ylabel("cascade fraction"); a1.legend(fontsize=8)
        a2.axvline(4.51, ls=":", color="k"); a2.set_title("GCC fraction (curves cross at $z_c$)")
        a2.set_xlabel("mean degree z"); a2.set_ylabel("giant-component fraction"); a2.legend(fontsize=8)
        a3.axvline(4.51, ls=":", color="k"); a3.set_title(r"susceptibility $\chi=N\,\mathrm{Var}(\Psi)$ (peak grows with N)")
        a3.set_xlabel("mean degree z"); a3.set_ylabel(r"$\chi$"); a3.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"{args.tag}.png"), dpi=120)
        print(f"wrote {args.tag}.png", flush=True)
    except ImportError:
        pass
    print(f"wrote {args.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
