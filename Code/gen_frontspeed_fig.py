#!/usr/bin/env python3
"""Fig: the recruitment SIGNAL outruns the BODIES (network<->space coupling, B.1).

Per tick, recruitment reach (distance a newly-active agent sits from the nearest previously-active agent)
vs agent displacement. Reach sits ~150x above displacement: coordination is carried by the percolating
network, not locomotion. Data: Code/sweeps/b1_frontspeed_raw.npz (near-critical run; regenerate with
b1_frontspeed.py). -> Code/figures/fig_frontspeed.pdf
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 10})
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "sweeps", "b1_frontspeed_raw.npz")
OUT = os.path.join(HERE, "figures", "fig_frontspeed.pdf")

d = np.load(NPZ)
va, reach, r = d["v_agent"], d["reach"], float(d["r"])
va = va[np.isfinite(va)]; reach = reach[np.isfinite(reach)]


def ccdf(x):
    x = np.sort(x[x > 0]); return x, 1.0 - np.arange(len(x)) / len(x)


fig, ax = plt.subplots(figsize=(5.4, 3.9))
xb, cb = ccdf(va); xs, cs = ccdf(reach)
ax.loglog(xb, cb, color="#c0392b", lw=2, label="agent displacement / tick (body)")
ax.loglog(xs, cs, color="#1250f0", lw=2, label="recruitment reach / tick (signal)")
mb, ms = np.median(va), np.median(reach)
ax.axvline(mb, color="#c0392b", ls=":", lw=1); ax.axvline(ms, color="#1250f0", ls=":", lw=1)
ax.axvline(r, color="0.4", ls="--", lw=1)
ax.text(r * 1.05, 0.5, "interaction\nradius $r$", color="0.4", fontsize=8, va="center")
ax.annotate("", xy=(ms, 0.04), xytext=(mb, 0.04), arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
ax.text(np.sqrt(mb * ms), 0.055, f"${ms/mb:.0f}\\times$", ha="center", fontsize=11, fontweight="bold")
ax.set_xlabel("distance per tick  (world units)")
ax.set_ylabel(r"$P(\geq \mathrm{distance})$")
ax.set_ylim(3e-4, 1.3)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout(pad=0.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)
print(f"median body={mb:.2f}  median signal={ms:.2f}  ratio={ms/mb:.1f}x  r={r:.1f}")
