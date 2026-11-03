#!/usr/bin/env python3
"""
A1 analysis + figure + VERDICT. Reads collapse_untuned.npz (and, for contrast, the TUNED anchors in
cascade_fss/fss3_uniform_N*.npz). Tests whether the UNTUNED (fixed-theta) cascade sizes collapse:
  - fit a single field-set tau from the pooled interior CCDF slopes,
  - plot the collapse s^{tau-1} P(S>=s) vs s/N for all N,
  - quantify overlay quality (spread of the rescaled curves) and whether the cutoff grows ~ N.
KEEP "self-organized (non-universal) criticality": curves overlay (low spread), cutoff ∝ N, stable tau.
DOWNGRADE to "near-critical band": no overlay / cutoff not scaling with N / tau drifts with N.
"""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

OUT = os.path.join(_P.DATA, "v10_revision")
FIGDIR = _P.FIG
d = np.load(os.path.join(OUT, "collapse_untuned.npz"), allow_pickle=True)
Ns = d["Ns"]; theta0 = float(d["theta0"])


def ccdf(sizes):
    s = np.sort(sizes[sizes >= 1]); n = len(s)
    xs, idx = np.unique(s, return_index=True)
    surv = 1.0 - idx / n
    return xs, surv


def fit_tau(sizes, N):                       # CCDF slope over the scaling window 2 <= s <= 0.5N
    xs, sv = ccdf(sizes)
    m = (xs >= 2) & (xs <= 0.5 * N)
    if m.sum() < 4:
        return np.nan
    sl = np.polyfit(np.log(xs[m]), np.log(sv[m]), 1)[0]
    return 1.0 - sl


taus_fit = np.array([fit_tau(d[f"sizes_N{N}"], N) for N in Ns])
tau = float(np.nanmedian(taus_fit))          # single field-set exponent for the collapse test

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2))
# (a) raw survival
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(Ns)))
for N, c in zip(Ns, cmap):
    xs, sv = ccdf(d[f"sizes_N{N}"]); ax[0].loglog(xs, sv, color=c, lw=1.4, label=f"N={N}")
# mean-field critical reference slope P(S>=s) ~ s^{-1/2} (tau=3/2), the tuned-critical exponent (Fig. S-fsscollapse):
# the untuned cascades track it through the scaling region, then truncate at the finite cutoff (near-critical, below).
xs0, sv0 = ccdf(d[f"sizes_N{Ns[-1]}"])
cref = sv0[np.argmin(np.abs(xs0 - 5.0))] * (5.0 ** 0.5)   # anchor to the largest-N curve near s=5
sref = np.array([2.0, 300.0])
ax[0].loglog(sref, cref * sref ** (-0.5), "k--", lw=1.4, label=r"$s^{-1/2}$ critical ($\tau=3/2$)")
ax[0].set_xlabel("cascade size s"); ax[0].set_ylabel(r"$P(S\geq s)$")
ax[0].set_title(f"(a) untuned survival ($\\phi$={theta0})")
ax[0].legend(fontsize=8, loc="lower left", framealpha=0.9, labelspacing=0.3, handlelength=1.3)
# (b) collapse s^{tau-1} P vs s/N
ovl = []
for N, c in zip(Ns, cmap):
    xs, sv = ccdf(d[f"sizes_N{N}"])
    ax[1].loglog(xs / N, (xs ** (tau - 1)) * sv, color=c, lw=1.4)  # N key shared with panel (a)
    m = (xs >= 2) & (xs <= 0.6 * N)          # collect rescaled values on a common x-grid for spread
    ovl.append((xs[m] / N, (xs[m] ** (tau - 1)) * sv[m]))
xg = np.logspace(-1.5, -0.1, 25)
stacks = []
for x, y in ovl:
    if len(x) > 3:
        stacks.append(np.interp(np.log(xg), np.log(x), np.log(y), left=np.nan, right=np.nan))
stacks = np.array(stacks)
with np.errstate(invalid="ignore"):
    spread = np.nanmean(np.nanstd(stacks, axis=0))   # mean log-spread of rescaled curves (0 = perfect collapse)
# same mean-field critical reference as (a), mapped into the collapse axes (rescaled by s^{tau-1}):
# y = s^{tau-1} * (cref * s^{-1/2}) = cref * s^{tau-3/2}, x = s/N (anchored to the largest N, as in (a)).
Nref = Ns[-1]
sb = np.array([2.0, 0.6 * Nref])
ax[1].loglog(sb / Nref, cref * sb ** (tau - 1.5), "k--", lw=1.4,
             label=r"$s^{-1/2}$ critical (rescaled)")
ax[1].set_xlabel("s / N"); ax[1].set_ylabel(r"$s^{\tau-1}P(S\geq s)$")
ax[1].set_title(f"(b) collapse test  ($\\tau$={tau:.2f}, spread={spread:.2f})")
ax[1].legend(fontsize=9, loc="lower left", framealpha=0.9)  # dashed reference only; N key in (a)
# (c) cutoff (max cascade) and tau vs N
maxsz = np.array([d[f"sizes_N{N}"].max() for N in Ns])
ax[2].loglog(Ns, maxsz, "o-", label="max cascade size")
ax[2].loglog(Ns, Ns, "k--", lw=1, label="∝ N (critical cutoff)")
ax2b = ax[2].twinx(); ax2b.semilogx(Ns, taus_fit, "s-", color="crimson", label=r"fitted $\tau$")
ax2b.set_ylabel(r"fitted $\tau$ per N", color="crimson"); ax2b.set_ylim(1.0, 2.6)
ax[2].set_xlabel("N"); ax[2].set_ylabel("max cascade size")
ax[2].set_title("(c) cutoff vs N  &  tau drift"); ax[2].legend(fontsize=12, loc="upper left")
fig.tight_layout(w_pad=0.4); fig.savefig(os.path.join(FIGDIR, "fig_untuned_collapse.png"), dpi=140)
fig.savefig(os.path.join(FIGDIR, "fig_untuned_collapse.pdf"))

# cutoff scaling exponent (max ~ N^p ; p~1 => critical)
p = np.polyfit(np.log(Ns), np.log(maxsz), 1)[0]
tau_drift = np.nanmax(taus_fit) - np.nanmin(taus_fit)
print(f"theta0={theta0}")
print(f"sigmas (untuned, biased est): {np.round(d['sigmas'],3)}")
print(f"fitted tau per N: {np.round(taus_fit,2)}  (drift {tau_drift:.2f})")
print(f"collapse log-spread = {spread:.3f}  (|<0.15 good collapse)")
print(f"cutoff scaling max~N^{p:.2f}  (p~1 => critical cutoff)")
good = (spread < 0.15) and (abs(p - 1) < 0.2) and (tau_drift < 0.25)
print("VERDICT:", "KEEP self-organized (non-universal) criticality (untuned sizes collapse, cutoff∝N, stable tau)"
      if good else "DOWNGRADE to 'parameter-set near-critical band' (no clean untuned collapse)")
print("A2 robustness: thetas", np.round(d["thetas"],2), "-> sigma", np.round(d["a2_sigma"],3),
      "tau", np.round(d["a2_tau"],2))
