#!/usr/bin/env python3
"""Fig 14 (SI, fig:homeostasis). (a) resource & exploring-fraction co-oscillation -- SMOOTHED for
viewability (faint raw + bold moving average). (b) corr(resource, exploring) positive across operating
points, collapsing under the recruitment->motion ablation. v14 output. CPU, deterministic (seed=0)."""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 12.5, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 11.5, "legend.fontsize": 11})
sys.path.insert(0, "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code")
from mobile_cascade import MobileCascade, MobileCascadeConfig


def series(converge=True, T=1000, seed=0, **kw):
    p = dict(rho_r=2.1, phi=0.02, theta=0.18, alpha=1.0, patch=40.0, clumps=6)
    p.update(kw)
    c = MobileCascadeConfig(B=24, N=120, K=15, L=800.0, rho_r=p["rho_r"], phi=p["phi"],
                            theta=p["theta"], levy_alpha=p["alpha"], clump_count=p["clumps"],
                            clump_radius=p["patch"], capacity=40.0, consumption=1.0, step=20.0,
                            pull=0.12, converge=converge, device="cpu", seed=seed)
    o = MobileCascade(c).run(T)
    return o["resource"].cpu().numpy()[:, 200:], o["exploring_fraction"].cpu().numpy()[:, 200:]


def corr(res, exp):
    cs = [np.corrcoef(res[b], exp[b])[0, 1] for b in range(res.shape[0]) if exp[b].std() > 1e-6]
    return (np.mean(cs), np.std(cs)) if cs else (np.nan, np.nan)


def smooth(y, w=11):
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")


ops = [("baseline", {}), ("α=0.7", dict(alpha=0.7)), ("α=1.6", dict(alpha=1.6)),
       ("φ sparse", dict(phi=0.005)), ("φ abundant", dict(phi=0.08)),
       ("θ=0.10", dict(theta=0.10)), ("θ=0.30", dict(theta=0.30)),
       ("ρ_r=1", dict(rho_r=1.0)), ("ρ_r=6", dict(rho_r=6.0)),
       ("tight", dict(patch=16.0)), ("diffuse", dict(patch=200.0))]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

# Left: co-oscillation, one trajectory -- faint raw + bold smoothed
res, exp = series()
r0, e0 = res[0] / res[0].max(), exp[0]; t = np.arange(len(r0))
ax[0].plot(t, r0, color="#c0392b", lw=0.6, alpha=0.25)
ax[0].plot(t, e0, color="#2980b9", lw=0.6, alpha=0.25)
ax[0].plot(t, smooth(r0), color="#c0392b", lw=1.8, label="resource (normalised)")
ax[0].plot(t, smooth(e0), color="#2980b9", lw=1.8, label="exploring fraction")
ax[0].set_xlim(0, 400); ax[0].set_xlabel("tick"); ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel("normalised level  (0–1)")
ax[0].legend(fontsize=11, loc="upper right")
ax[0].set_title("(a)", loc="left", fontweight="bold", fontsize=13)

# Right: robustness + ablation
labels, means, stds = [], [], []
for lab, kw in ops:
    m, s = corr(*series(converge=True, **kw)); labels.append(lab); means.append(m); stds.append(s)
abl_m, abl_s = corr(*series(converge=False))
y = np.arange(len(labels))
ax[1].errorbar(means, y, xerr=stds, fmt="o", color="#27ae60", capsize=2, ms=5, label="recruiting")
ax[1].axvline(0, color="k", lw=1, ls="--")
ax[1].axvline(abl_m, color="#c0392b", lw=1.4, ls=":", label=f"ablated (off): {abl_m:+.2f}")
ax[1].set_yticks(y); ax[1].set_yticklabels(labels, fontsize=10.5)
ax[1].set_ylabel("operating point"); ax[1].set_xlabel("corr(resource, exploring fraction)")
ax[1].set_xlim(-0.22, 0.45)
ax[1].legend(fontsize=9.5, loc="lower left", framealpha=0.95)
ax[1].set_title("(b)", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout()
out = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/theory_paper_v14/figures/fig_homeostasis.pdf"
fig.savefig(out)
print("wrote", out)
print("recruiting corr range: %.2f..%.2f ; ablated %.2f" % (min(means), max(means), abl_m))
