#!/usr/bin/env python3
"""SI panel: sparsity amplifies the effective clustering strength beta_eff.

Effective (dilute-limit) clustering strength of a point field = the max-entropy multiplier that
would reproduce its close-pair count, estimated by the second-virial relation
    beta_eff = log( <C>_obs / <C>_CSR ),
with <C>_CSR the close pairs a CSR (Poisson) field of the SAME density would give.

Claim (derivation in SI): for a renewal/cluster field with fixed clump occupancy mu,
    beta_eff ~ log(mu) - log(N) - log(pi r0^2)   =>   grows like log(1/density), slope +1.
The two max-ent arguments multiply: the counting argument sets the CSR denominator (~density, -> 0 when
sparse); autocatalytic renewal sets the numerator (~N*mu, offspring beside parents). So the sparsity that
foraging presupposes amplifies the contrast recruitment exploits.

Illustrated on a Thomas (Neyman-Scott) cluster process -- NOT literally e^{beta C}; beta_eff is the
descriptive max-ent fit. Sweep mu to show the SLOPE is mu-independent (the real test of the derivation);
the intercept shifts by log(mu). Pure numpy, torus [0,1]^2. -> figures/fig_beta_sparsity.pdf
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "fig_beta_sparsity.pdf")
rng = np.random.default_rng(0)

SIGC = 0.012       # clump spatial scale (offspring spread about parent)
R0   = 0.025       # close-pair counting range (>= SIGC so within-clump pairs count)
REPS = 24          # realizations per (mu, density)
MUS  = [10, 20, 40]
NP_LIST = [4, 8, 16, 32, 64, 128]   # parents -> density N = n_par * mu
COLORS = {10: "#2c7fb8", 20: "#c0392b", 40: "#31843f"}


def minimg(d):
    return d - np.round(d)


def thomas(n_par, mu):
    par = rng.random((n_par, 2))
    off = par[:, None, :] + SIGC * rng.standard_normal((n_par, mu, 2))
    return (off.reshape(-1, 2)) % 1.0


def close_pairs(pts, r0):
    n = len(pts)
    d = minimg(pts[:, None, :] - pts[None, :, :])
    r2 = (d ** 2).sum(2)
    iu = np.triu_indices(n, 1)
    return int((r2[iu] < r0 ** 2).sum())


def beta_eff(pts):
    N = len(pts)
    c_obs = close_pairs(pts, R0)
    c_csr = N * (N - 1) / 2.0 * (np.pi * R0 ** 2)
    return np.log(max(c_obs, 1) / c_csr)


fig, ax = plt.subplots(figsize=(5.4, 4.2))
print(f"{'mu':>4} {'N':>6} {'logN':>7} {'beta_eff':>9} {'sd':>6}")
fit_lines = []
for mu in MUS:
    xs, ys, yerr = [], [], []
    for n_par in NP_LIST:
        N = n_par * mu
        b = np.array([beta_eff(thomas(n_par, mu)) for _ in range(REPS)])
        xs.append(np.log(1.0 / N)); ys.append(b.mean()); yerr.append(b.std())
        print(f"{mu:>4} {N:>6} {np.log(N):>7.3f} {b.mean():>9.3f} {b.std():>6.3f}")
    xs, ys, yerr = map(np.array, (xs, ys, yerr))
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intc = np.linalg.lstsq(A, ys, rcond=None)[0]
    fit_lines.append((mu, slope, intc))
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", ms=5, color=COLORS[mu], capsize=2,
                label=rf"$\mu={mu}$  (slope ${slope:.2f}$)")
    # analytic prediction: beta_eff = log(mu) - log(N) - log(pi r0^2) = log(mu) - log(pi r0^2) + x
    pred = np.log(mu) - np.log(np.pi * R0 ** 2) + xs
    ax.plot(xs, pred, "--", color=COLORS[mu], lw=1.2, alpha=0.7)

ax.plot([], [], "k--", lw=1.2, alpha=0.7, label="analytic (slope 1)")
ax.set_xlabel(r"$\log(1/\mathrm{density})$  (sparser $\rightarrow$)")
ax.set_ylabel(r"effective clustering strength $\beta_{\rm eff}=\log(\langle\mathcal{C}\rangle/\langle\mathcal{C}\rangle_{\rm CSR})$")
ax.set_title("Sparsity amplifies the effective clustering strength", fontsize=11)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT); print("\nwrote", os.path.normpath(OUT))
print("slopes (predicted 1.0, mu-independent):", [f"mu={m}:{s:.3f}" for m, s, _ in fit_lines])
