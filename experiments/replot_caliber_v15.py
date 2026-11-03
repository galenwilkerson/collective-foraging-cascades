#!/usr/bin/env python3
"""CPU replot of Fig 6 (fig_caliber_opt) from the engine-independent caliber results.npz -- figure polish
only, no re-run. The single-searcher caliber sweep is engine-independent, so the v14 results.npz is reused.
Fix: panel (a) legend moved to lower left with transparency (was lower center, overlaying the curves)."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
RESULTS = os.path.join(_P.DATA, "maxent_caliber/results.npz")
FIGDIR = _P.FIG
d = np.load(RESULTS, allow_pickle=True)
mu = d["alphas"]; astab = mu - 1.0
eta = d["eta"]; eta_sd = d["eta_sd"]; Dfs = d["Dfs"]; betas = d["betas"]
P = d["params"].item(); nrep = P["RF"] * P["KS"]


def smooth(y, k=1):
    out = y.copy()
    for i in range(len(y)):
        lo, hi = max(0, i - k), min(len(y), i + k + 1); out[i] = y[lo:hi].mean()
    return out


def alpha_star(e):
    es = smooth(e); w = np.clip(es - 0.85 * es.max(), 0, None)
    return float((w * astab).sum() / w.sum()) if w.sum() > 0 else float(astab[np.argmax(es)])


astar = np.array([alpha_star(eta[bi]) for bi in range(len(betas))])
cc = np.corrcoef(Dfs, astar)[0, 1]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.5))
order = np.argsort(Dfs)
pick = [order[0], order[len(order) // 2], order[-1]]
labels = ["strongly clustered", "intermediate", "diffuse"]
cols = ["#5e3c99", "#1a9641", "#e6801a"]; marks = ["o", "s", "^"]
for c, m, lab, k in zip(cols, marks, labels, pick):
    peak = eta[k].max(); e = smooth(eta[k]) / peak; sem = (eta_sd[k] / np.sqrt(nrep)) / peak
    axA.fill_between(astab, e - sem, e + sem, color=c, alpha=0.18, lw=0)
    axA.plot(astab, e, "-", color=c, lw=1.8, zorder=3)
    axA.plot(astab, e, m, color=c, ms=6, mec="white", mew=0.6, zorder=4,
             label=f"{lab} (D$_f$={Dfs[k]:.2f}, α*≈{astar[k]:.2f})")
    pk = int(np.argmax(smooth(eta[k])))
    axA.plot(astab[pk], e[pk], "*", color=c, ms=17, mec="k", mew=0.7, zorder=6)
axA.plot([], [], "k*", ms=11, label="optimum α* (shifts left as clustering grows)")
axA.set_xlabel(r"search — Lévy stable index $\alpha$  (ballistic $\leftarrow\alpha\rightarrow$ Brownian)")
axA.set_ylabel(r"foraging efficiency $\eta$ (normalised)")
axA.set_title("(a) efficiency vs search exponent"); axA.set_ylim(0.25, 1.08)
axA.legend(fontsize=8, loc="lower left", framealpha=0.80)

lo = min(Dfs.min(), astar.min()) - 0.05; hi = max(Dfs.max(), astar.max()) + 0.05
axB.plot([lo, hi], [lo, hi], "--", color="0.55", lw=1.2, zorder=1, label=r"identity $\alpha^*=D_f$")
axB.plot(Dfs, astar, "o", color="#1250f0", ms=8, zorder=3, label=r"measured $\alpha^*$")
axB.set_xlabel(r"resource-field fractal dimension $D_f$  (clustered $\leftarrow\rightarrow$ diffuse)")
axB.set_ylabel(r"optimal search exponent $\alpha^*$")
axB.set_title(r"(b) optimal exponent rises with $D_f$  ($\alpha^*\!\approx\!D_f$)")
axB.set_xlim(lo, hi); axB.set_ylim(lo, hi); axB.set_aspect("equal", "box")
axB.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_caliber_opt.pdf"))
print("wrote fig_caliber_opt.pdf | corr(alpha*,D_f)=%.2f" % cc)
