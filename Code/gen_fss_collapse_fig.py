#!/usr/bin/env python3
"""Fig: finite-size scaling of recruitment-cascade sizes on the generic (max-ent/CSR) field.

At matched criticality (sigma_local~1, theta calibrated per N) the cascade-size distributions on the
generic sparse field collapse onto the mean-field critical-branching form P(S>=s) ~ s^{-1/2} G(s/N)
(avalanche exponent tau=3/2, cutoff at swarm size N). Data: Code/sweeps/fss3_uniform_N*.npz (regenerate
with run_sigma_matched.py). -> Code/figures/fig_fss_collapse.pdf
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 9})
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sweeps")
OUT = os.path.join(HERE, "figures", "fig_fss_collapse.pdf")
NS = [50, 100, 200, 400, 800]
TAU = 1.5


def surv(s, smin=2):
    s = np.sort(s[s >= smin].astype(float)); return s, 1.0 - np.arange(len(s)) / len(s)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
cmap = plt.cm.viridis(np.linspace(0, 0.88, len(NS)))
sig = {}
for N, c in zip(NS, cmap):
    d = np.load(os.path.join(DATA, f"fss3_uniform_N{N}.npz"))
    sig[N] = float(d["sigma_aval"][0])
    s, cc = surv(d["sizes"])
    a1.loglog(s, cc, color=c, lw=1.6, label=f"$N={N}$")
    a2.loglog(s / N, cc * s ** (TAU - 1), color=c, lw=1.6, label=f"$N={N}$ ($\\sigma{{=}}{sig[N]:.2f}$)")
sref = np.array([2, 700]); a1.loglog(sref, 1.2 * (sref / 2.0) ** -0.5, "k--", lw=1.2, zorder=1,
                                     label=r"$s^{-1/2}$ ($\tau{=}3/2$)")
a1.set_xlabel("cascade size $s$"); a1.set_ylabel(r"$P(S\geq s)$"); a1.legend(frameon=False, loc="lower left")
a1.set_title("(a) cascade-size distributions", loc="left", fontsize=11)
a2.axhline(1.2, color="k", ls="--", lw=1.0, zorder=1)
a2.set_xlabel(r"$s/N$"); a2.set_ylabel(r"$s^{1/2}\,P(S\geq s)$")
a2.set_title(r"(b) collapse: mean-field branching $\tau=3/2$", loc="left", fontsize=11)
a2.legend(frameon=False, loc="lower left")
fig.tight_layout(pad=0.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)
print("sigma_local(N):", {N: round(sig[N], 3) for N in NS})
