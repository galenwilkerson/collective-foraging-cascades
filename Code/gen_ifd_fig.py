#!/usr/bin/env python3
"""Generate fig_ifd.pdf from the saved ifd_test.npz (produced by ifd_test.py).

Plots log10(active agents at clump) vs log10(patch richness) for the recruiting (coupling ON) and
ablated (coupling OFF) conditions, with the fitted matching slope and the slope-one IFD reference.
Recruiting reaches slope ~0.84 (near-ideal, mild undermatching); ablation collapses to ~0.09.
No suptitle -- the LaTeX caption carries the interpretation.  ->  theory_paper_v3/figures/fig_ifd.pdf
"""
import os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5})

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
d = np.load(os.path.join(SW, "ifd_test.npz"))


def fit(rich, occ):
    r, o = rich.flatten(), occ.flatten()
    m = (r >= 1) & (o >= 0.5)
    x, y = np.log10(r[m]), np.log10(o[m])
    A = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - np.polyval(A, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return x, y, A, r2


fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for k, (rk, ok, col, title) in enumerate([
        ("rich_on", "occ_on", "#2ca02c", "recruiting (coupling on)"),
        ("rich_off", "occ_off", "#c0392b", "ablated (coupling off)")]):
    x, y, A, r2 = fit(d[rk], d[ok])
    ax[k].scatter(x, y, s=14, color=col, alpha=0.5, edgecolors="none")
    xs = np.linspace(x.min(), x.max(), 50)
    ax[k].plot(xs, np.polyval(A, xs), color=col, lw=2, label=f"slope {A[0]:.2f}")
    ax[k].plot(xs, xs + (y.mean() - x.mean()), "k--", lw=1, label="IFD (slope 1)")
    ax[k].set_title(f"({'ab'[k]})", loc="left", fontweight="bold", fontsize=13)
    ax[k].set_xlabel(r"$\log_{10}$ patch richness (sources/clump)")
    ax[k].legend(fontsize=11.5, loc="upper left", frameon=False)
    print(f"{title}: slope={A[0]:.2f} R2={r2:.2f}")
ax[0].set_ylabel(r"$\log_{10}$ active agents at clump")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory_paper_v3", "figures", "fig_ifd.pdf")
fig.savefig(out); print("wrote", os.path.normpath(out))
