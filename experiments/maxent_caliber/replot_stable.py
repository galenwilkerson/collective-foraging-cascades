#!/usr/bin/env python3
"""Clean two-panel SI figure from results.npz, in the paper's Lévy convention (stable index alpha = mu-1).
(a) eta(alpha) for three representative fields (clustered / medium / diffuse) with +/-SEM bands and markers.
(b) robust optimal alpha* vs field fractal dimension D_f -- the summary trend.
alpha* = centre-of-mass of the near-optimal region (>=85% of peak), robust to single-point noise."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "results.npz"), allow_pickle=True)
mu = d["alphas"]; astab = mu - 1.0
eta = d["eta"]; eta_sd = d["eta_sd"]; Dfs = d["Dfs"]; betas = d["betas"]
P = d["params"].item(); nrep = P["RF"] * P["KS"]

def smooth(y, k=1):
    out = y.copy()
    for i in range(len(y)):
        lo, hi = max(0, i-k), min(len(y), i+k+1); out[i] = y[lo:hi].mean()
    return out

def alpha_star(e):                       # centre-of-mass of the near-optimal (>=85% peak) region
    es = smooth(e); w = np.clip(es - 0.85*es.max(), 0, None)
    return float((w*astab).sum()/w.sum()) if w.sum() > 0 else float(astab[np.argmax(es)])

astar = np.array([alpha_star(eta[bi]) for bi in range(len(betas))])
cc = np.corrcoef(Dfs, astar)[0, 1]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.5))

# --- (a) three representative fields ---
order = np.argsort(Dfs)
pick = [order[0], order[len(order)//2], order[-1]]          # most clustered / medium / diffuse
labels = ["strongly clustered", "intermediate", "diffuse"]
cols = ["#5e3c99", "#1a9641", "#e6801a"]; marks = ["o", "s", "^"]
for c, m, lab, k in zip(cols, marks, labels, pick):
    peak = eta[k].max(); e = smooth(eta[k])/peak; sem = (eta_sd[k]/np.sqrt(nrep))/peak
    axA.fill_between(astab, e-sem, e+sem, color=c, alpha=0.18, lw=0)
    axA.plot(astab, e, "-", color=c, lw=1.8, zorder=3)
    axA.plot(astab, e, m, color=c, ms=6, mec="white", mew=0.6, zorder=4,
             label=f"{lab} (D$_f$={Dfs[k]:.2f}, α*≈{astar[k]:.2f})")
    pk = int(np.argmax(smooth(eta[k])))                       # mark each field's OWN optimum
    axA.plot(astab[pk], e[pk], "*", color=c, ms=17, mec="k", mew=0.7, zorder=6)
axA.plot([], [], "k*", ms=11, label="optimum α* (shifts left as clustering grows)")
axA.set_xlabel(r"search — Lévy stable index $\alpha$  (ballistic $\leftarrow\alpha\rightarrow$ Brownian)")
axA.set_ylabel(r"foraging efficiency $\eta$ (normalised)")
axA.set_title("(a) efficiency vs search exponent"); axA.set_ylim(0.25, 1.08)
axA.legend(fontsize=8, loc="lower center")

# --- (b) alpha* vs D_f trend: near the identity line, i.e. the search exponent tracks the field ---
lo = min(Dfs.min(), astar.min()) - 0.05; hi = max(Dfs.max(), astar.max()) + 0.05
axB.plot([lo, hi], [lo, hi], "--", color="0.55", lw=1.2, zorder=1, label=r"identity $\alpha^*=D_f$")
axB.plot(Dfs, astar, "o", color="#1250f0", ms=8, zorder=3, label=r"measured $\alpha^*$")
axB.set_xlabel(r"resource-field fractal dimension $D_f$  (clustered $\leftarrow\rightarrow$ diffuse)")
axB.set_ylabel(r"optimal search exponent $\alpha^*$")
axB.set_title(f"(b) the search exponent tracks the field  (r={cc:.2f})")
axB.set_xlim(lo, hi); axB.set_ylim(lo, hi); axB.set_aspect("equal", "box")
axB.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_caliber_opt.pdf"))
fig.savefig(os.path.join(HERE, "fig_caliber_opt.png"), dpi=140)
print("alpha* by D_f:", {round(float(Dfs[k]),2): round(float(astar[k]),2) for k in order})
print(f"corr(alpha*, D_f) = {cc:.2f}   nrep/point = {nrep}")
