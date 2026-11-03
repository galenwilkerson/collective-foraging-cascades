#!/usr/bin/env python3
"""Fig_frontspeed (v15): recruitment signal outruns the bodies. Plots the CCDFs of per-tick agent
displacement (body) vs recruitment reach (signal) from the corrected-engine run
(b1_frontspeed.py -> b1_frontspeed_raw.npz). Plotter only; adapted from v8 with the v15 output path."""
import os, numpy as np, matplotlib
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 17, "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "legend.fontsize": 15})
NPZ = os.path.join(_P.DATA, "cascade_fss/b1_frontspeed_raw.npz")
OUT = os.path.join(_P.FIG, "fig_frontspeed.pdf")
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
ax.text(r * 0.9, 0.012, "interaction\nradius $r$", color="0.4", fontsize=12, va="center", ha="right")
ax.annotate("", xy=(ms, 0.04), xytext=(mb, 0.04), arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
ax.text(np.sqrt(mb * ms), 0.055, f"${ms/mb:.0f}\\times$", ha="center", fontsize=15, fontweight="bold")
ax.set_xlabel("distance per tick  (world units)")
ax.set_ylabel(r"$P(\geq \mathrm{distance})$")
ax.set_ylim(3e-4, 1.3)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout(pad=0.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
print(f"wrote {OUT} | median body={mb:.2f} signal={ms:.2f} ratio={ms/mb:.0f}x r={r:.1f}")
