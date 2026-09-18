"""Fig: recruitment-CASCADE avalanche-size distribution (connected-component size of each branching
cascade tree) on the running foraging model, Levy a=1.5. Two thresholds: phi=0.20 (sigma_local~0.99,
near-critical) vs phi=0.35 (sigma_local~0.83, subcritical). Overlaid with the mean-field
critical-branching reference P(S>=s) ~ s^(-1/2) (avalanche PDF tau=3/2). The exponent tracks
sigma_local; finite-size cutoff at s=N limits its precision, which is why sigma_local (not this slope)
is the primary criticality measure. Data: Code/sweeps/local_branching_levy.npz (canonical alpha=1.5)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NPZ = os.path.join(REPO, "Code", "sweeps", "local_branching_levy.npz")
OUT = os.path.join(REPO, "theory_paper_v14", "figures", "fig_recruit_cluster.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = np.load(NPZ)
N = int(d["N"][0])
FITLO, FITHI = 3, 50   # scaling window: above the s=2 head, below the finite-size (s->N) pile-up


def survival(s):
    u, cnt = np.unique(s[s >= 2].astype(float), return_counts=True)
    return u, cnt[::-1].cumsum()[::-1] / cnt.sum()


def slope(u, c):
    m = (u >= FITLO) & (u <= FITHI) & (c > 0)
    b = np.polyfit(np.log10(u[m]), np.log10(c[m]), 1)
    return b[0]


fig, ax = plt.subplots(figsize=(5.2, 3.9))
info = [("0.20", "#1250f0", "o", 0.99), ("0.35", "#d1495b", "s", 0.83)]
for phi, col, mk, sig in info:
    u, c = survival(d["sizes_" + phi])
    sl = slope(u, c); tau = 1 - sl
    ax.loglog(u, c, mk, ms=3.4, mfc=col if mk == "o" else "none", mec=col, mew=1.0,
              color=col, alpha=0.7, rasterized=True,
              label=fr"$\phi{{=}}{phi}$ ($\sigma_{{\rm local}}{{\approx}}{sig}$): $\tau{{\approx}}{tau:.1f}$")

# mean-field critical-branching reference: CCDF ~ s^{-1/2} (avalanche PDF tau=3/2), anchored at s=2
sref = np.array([2, 0.7 * N]); yref = 1.0 * (sref / 2.0) ** (-0.5)
ax.loglog(sref, yref, "--", color="0.35", lw=1.4, zorder=1,
          label=r"critical branching $\propto s^{-1/2}$ ($\tau{=}3/2$)")
ax.axvline(N, color="0.45", lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.text(N * 0.96, 1.1, r"finite-size cutoff $s{=}N$", ha="right", va="bottom", fontsize=8, color="0.4")
ax.set_xlabel(r"cascade (connected-component) size $s$")
ax.set_ylabel(r"$P(S \geq s)$")
ax.set_ylim(2e-4, 1.6)
ax.legend(fontsize=8, frameon=False, loc="lower left", handletextpad=0.5, borderaxespad=0.3)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=300)
for phi, col, mk, sig in info:
    u, c = survival(d["sizes_" + phi]); sl = slope(u, c)
    print(f"phi={phi}: sigma_local~{sig}  CCDF slope[{FITLO},{FITHI}]={sl:.2f}  tau={1-sl:.2f}  "
          f"n={len(d['sizes_'+phi])} max={int(d['sizes_'+phi].max())}")
print("wrote", OUT)
