#!/usr/bin/env python3
"""fig_susceptibility (v15): local cluster-size susceptibility chi=<S^2>/<S> (largest cluster excluded)
vs mean degree z, on the CORRECTED engine with Levy exploration (the model's actual exploration mode).
Data: /mnt/newssd/.../local_susceptibility_levy.npz (gen_susceptibility_cuda.py --levy 1.5). Plotter only.
Shaded band: 95% bootstrap CI over replicas (chi_reps)."""
import os, numpy as np, matplotlib
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11})
NPZ = os.path.join(_P.DATA, "v15_susc/local_susceptibility_levy.npz")
OUT = os.path.join(_P.FIG, "fig_susceptibility.pdf")
d = np.load(NPZ)
z = d["z_global"]; chi = d["chi"]; reps = d["chi_reps"]       # reps: (n_z, B)
rng = np.random.default_rng(0)
lo, hi = np.full_like(chi, np.nan), np.full_like(chi, np.nan)
for i in range(len(z)):
    v = reps[i][np.isfinite(reps[i])]
    if len(v) > 2:
        bs = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
        lo[i], hi[i] = np.percentile(bs, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.fill_between(z, lo, hi, color="#8e44ad", alpha=0.2, lw=0)
ax.plot(z, chi, "o-", color="#8e44ad", ms=6, lw=2,
        label=r"cluster susceptibility $\chi=\langle S^2\rangle/\langle S\rangle$")
pk = int(np.nanargmax(chi))
ax.axvline(z[pk], color="0.5", ls=":", lw=1)
ax.axvspan(4.51, 10.0, color="0.85", alpha=0.4, zorder=0)
ax.text(7.2, ax.get_ylim()[1] * 0.1, "cascade\nwindow", fontsize=10, color="0.4", ha="left")
ax.text(z[pk] + 0.5, chi[pk] * 0.92, f"peak at\n$z\\approx{z[pk]:.1f}$", fontsize=11)
ax.set_xlabel(r"mean degree $z$ (global)"); ax.set_ylabel(r"local susceptibility $\chi$")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=140)
print(f"wrote {OUT} | peak z={z[pk]:.2f} chi={chi[pk]:.2f} CI[{lo[pk]:.2f},{hi[pk]:.2f}]")
