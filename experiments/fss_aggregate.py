#!/usr/bin/env python3
"""
Aggregate the Anvil fss cells -> TUNED cascade-size FSS collapse figure (RQ2, Fig fig_fss_collapse).
For each N: over the theta grid, find theta* where the Galton-Watson sigma_aval crosses 1 (interp), take
that cell's cascade sizes, fit tau on the CCDF, and test the data collapse s^{tau-1} P(S>=s) vs s/N across
N. Honest verdict: a tight overlay (small log-spread) => clean FSS/SOC; a band => near-critical / SOqC.
Reads cell_N{N}_th{theta}.npz (written by fss_cell.py) from CELLDIR (rsync of Anvil out/), writes
v15/figures/fig_fss_collapse.pdf and prints theta*(N), tau(N), and the collapse log-spread.

Usage: FSS_CELLS=/path/to/cells python3 fss_aggregate.py
"""
import os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CELLDIR = os.environ.get("FSS_CELLS", os.path.join(HERE, "fss_cells"))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")


def ccdf(sizes):
    s = np.sort(np.asarray(sizes, dtype=float))
    s = s[s >= 1]
    if len(s) == 0:
        return np.array([]), np.array([])
    x, first = np.unique(s, return_index=True)
    return x, 1.0 - first / len(s)              # P(S >= x)


def tau_fit(x, c, N):
    m = (x >= 2) & (x <= 0.5 * N) & (c > 0)      # fit the scaling region, below the N-cutoff
    if m.sum() < 5:
        return np.nan
    slope = np.polyfit(np.log(x[m]), np.log(c[m]), 1)[0]
    return 1.0 - slope                           # P(S>=s) ~ s^{-(tau-1)}  => CCDF slope = -(tau-1)


def main():
    cells = {}
    for f in sorted(glob.glob(os.path.join(CELLDIR, "cell_N*_th*.npz"))):
        d = np.load(f, allow_pickle=True)
        if int(d["done_chunks"]) < int(d["chunks"]):
            print(f"  (incomplete, skipping) {os.path.basename(f)}: "
                  f"{int(d['done_chunks'])}/{int(d['chunks'])} chunks")
            continue
        cells.setdefault(int(d["N"]), []).append(
            (float(d["theta"]), float(d["sigma_aval"]), np.asarray(d["sizes"], float)))
    if not cells:
        print(f"no complete cells in {CELLDIR}"); return
    Ns = sorted(cells)
    theta_star, sizes_N, tau_N = {}, {}, {}
    for N in Ns:
        arr = sorted(cells[N])                   # by theta
        th = np.array([a[0] for a in arr]); sg = np.array([a[1] for a in arr])
        # sigma_aval decreases as theta rises; theta* = crossing of sigma=1
        if sg.min() < 1.0 < sg.max():
            order = np.argsort(sg)
            theta_star[N] = float(np.interp(1.0, sg[order], th[order]))
        else:
            theta_star[N] = float(th[np.argmin(np.abs(sg - 1.0))])
        j = int(np.argmin(np.abs(th - theta_star[N])))
        sizes_N[N] = arr[j][2]
        x, c = ccdf(sizes_N[N]); tau_N[N] = tau_fit(x, c, N)
    tau = float(np.nanmedian([t for t in tau_N.values() if t == t]))

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(Ns)))
    overlays, xg = [], np.logspace(-1.6, -0.15, 24)
    for N, col in zip(Ns, cols):
        x, c = ccdf(sizes_N[N])
        if len(x) == 0:
            continue
        ax[0].loglog(x, c, color=col, lw=1.3, label=f"N={N}")
        xs, ys = x / N, (x ** (tau - 1)) * c
        ax[1].loglog(xs, ys, color=col, lw=1.3, label=f"N={N}")
        m = (x >= 2) & (x <= 0.6 * N)
        if m.sum() > 3:
            overlays.append(np.interp(np.log(xg), np.log(xs[m]), np.log(ys[m]),
                                      left=np.nan, right=np.nan))
    spread = float(np.nanmean(np.nanstd(np.array(overlays), axis=0))) if overlays else np.nan
    ax[0].set_xlabel("cascade size $s$"); ax[0].set_ylabel("$P(S\\geq s)$")
    ax[0].set_title(f"tuned cascade sizes ($\\tau={tau:.2f}$)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("$s/N$"); ax[1].set_ylabel("$s^{\\tau-1}P(S\\geq s)$")
    verdict = "clean collapse" if spread < 0.15 else "near-critical band (SOqC)"
    ax[1].set_title(f"FSS collapse: {verdict} (spread={spread:.2f})")
    fig.tight_layout(); os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "fig_fss_collapse.pdf"); fig.savefig(out)
    print(f"theta*(N): " + ", ".join(f"{N}:{theta_star[N]:.3f}" for N in Ns))
    print(f"tau(N):    " + ", ".join(f"{N}:{tau_N[N]:.2f}" for N in Ns))
    print(f"median tau={tau:.2f}  collapse log-spread={spread:.3f} -> {verdict}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
