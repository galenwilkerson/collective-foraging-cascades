#!/usr/bin/env python3
"""Two maximum entropies: does the max-caliber (Lévy) search exponent that is optimal for foraging track the
maximum-entropy (scale-free) resource field's exponent?  See PLAN.md.

Environment  = Lévy-dust point field with step exponent beta (fractal dim D_f ~ beta-1).
Searcher     = Lévy walk with exponent alpha; non-destructive encounters counted per unit path length.
Output       = fig_eta_vs_alpha.pdf, fig_alphastar_vs_beta.pdf, results.npz, results.md
CPU, minutes.  Reproducible via SEED0.
"""
import os, numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEED0 = 20260908
L      = 1024.0          # torus side
M      = 500             # targets
RV     = 6.0             # detection radius = min step length
HOPS   = 3000            # hops per searcher run (fixed effort; efficiency normalises by actual path length)
BETAS  = np.array([1.3, 1.6, 1.9, 2.2, 2.5, 2.8])   # field exponents
ALPHAS = np.round(np.linspace(1.1, 3.0, 13), 3)     # search exponents (finer grid)
RF     = 12              # field realisations per beta
KS     = 16              # searcher replicas per (field, alpha)

def powerlaw_steps(n, mu, lmin, rng, lmax):
    """n step lengths with P(l) ~ l^-mu on [lmin, inf), inverse-transform; capped at lmax."""
    u = rng.random(n)
    l = lmin * u ** (-1.0 / (mu - 1.0))
    return np.minimum(l, lmax)

def levy_dust(beta, rng):
    """Targets = sites of a Lévy walk with exponent beta in the UNBOUNDED plane (preserves fractal structure,
    D_f ~ beta-1), then rescaled to fill a fixed box [0,L]^2 (scale-invariant clustering preserved, density fixed)."""
    ang = rng.random(M) * 2 * np.pi
    step = powerlaw_steps(M, beta, 1.0, rng, 1e9)       # no cap, no wrap -> keep the fractal
    x = np.cumsum(step * np.cos(ang)); y = np.cumsum(step * np.sin(ang))
    x -= x.min(); y -= y.min()
    span = max(x.max(), y.max(), 1e-9)
    x = 0.04 * L + 0.92 * L * x / span
    y = 0.04 * L + 0.92 * L * y / span
    return np.column_stack([x, y])

def box_count_dim(pts):
    """Fractal dimension by box counting on the torus."""
    scales = np.array([4, 8, 16, 32, 64, 128])
    counts = []
    for s in scales:
        cell = np.floor(pts / (L / s)).astype(int)
        counts.append(len({(a, b) for a, b in cell}))
    counts = np.array(counts, float)
    # slope of log N vs log s
    A = np.polyfit(np.log(scales), np.log(counts), 1)
    return A[0]

def dist_ccdf_slope(pts, rng):
    """CCDF slope of a sample of pairwise toroidal distances (log-log tail fit)."""
    idx = rng.integers(0, len(pts), size=(4000, 2))
    d = pts[idx[:, 0]] - pts[idx[:, 1]]
    r = np.sqrt((d ** 2).sum(1)); r = r[r > 0]
    r = np.sort(r); n = len(r); p = 1 - np.arange(n) / n
    m = np.median(r); mask = (r >= m) & (p > 0.01)          # fit the tail
    if mask.sum() < 10: return np.nan
    A = np.polyfit(np.log(r[mask]), np.log(p[mask]), 1)
    return A[0]

def reflect(v):
    """Reflect coordinate(s) into [0,L] (bounded domain, no periodicity)."""
    m = np.mod(v, 2 * L)
    return np.where(m > L, 2 * L - m, m)

def search_efficiency(tree, ntarg, alpha, rng):
    """Lévy searcher, DESTRUCTIVE foraging: targets consumed on first encounter. Returns targets found per
    unit path length. The walk stops early once most targets are gone (so idle post-depletion path doesn't
    dominate). This is the classic Viswanathan efficiency where heavy tails matter in sparse/clustered fields."""
    ang = rng.random(HOPS) * 2 * np.pi
    step = powerlaw_steps(HOPS, alpha, RV, rng, L)
    x = rng.random() * L; y = rng.random() * L
    alive = np.ones(ntarg, bool)
    found = 0; path = 0.0
    for k in range(HOPS):
        d = step[k]; path += d
        nx = x + d * np.cos(ang[k]); ny = y + d * np.sin(ang[k])
        ns = max(1, int(d / RV))                            # sample the hop segment every RV
        ts = np.linspace(0, 1, ns + 1)
        sx = reflect(x + (nx - x) * ts); sy = reflect(y + (ny - y) * ts)
        hit = tree.query_ball_point(np.column_stack([sx, sy]), RV)
        seen = set()
        for f in hit: seen.update(f)
        for t in seen:
            if alive[t]: alive[t] = False; found += 1
        x, y = float(reflect(nx)), float(reflect(ny))
        if alive.sum() < 0.15 * ntarg: break               # most targets consumed -> stop
    return found / path if path > 0 else 0.0

def field_job(arg):
    """One field realisation: build it, measure D_f + gap slope, run all (alpha, searcher) sweeps.
    Returned to the parent for aggregation. Module-level so multiprocessing can pickle it."""
    bi, rf = arg; beta = BETAS[bi]
    rng = np.random.default_rng(SEED0 + 1000 * bi + rf)
    pts = levy_dust(beta, rng)
    tree = cKDTree(pts)
    Df = box_count_dim(pts); gap = dist_ccdf_slope(pts, rng)
    per_alpha = []
    for ai, al in enumerate(ALPHAS):
        vals = []
        for ks in range(KS):
            rs = np.random.default_rng(SEED0 + 7 * (1000 * bi + rf) + 13 * ai + ks)
            vals.append(search_efficiency(tree, len(pts), al, rs))
        per_alpha.append(vals)
    return (bi, rf, Df, gap, per_alpha)

def main():
    from multiprocessing import Pool
    eta = np.zeros((len(BETAS), len(ALPHAS)))
    eta_sd = np.zeros_like(eta)
    Dfs = np.zeros(len(BETAS)); Dfs_sd = np.zeros(len(BETAS))
    gap = np.zeros(len(BETAS))
    work = [(bi, rf) for bi in range(len(BETAS)) for rf in range(RF)]
    nproc = min(len(work), os.cpu_count() or 4)
    print(f"running {len(work)} field jobs on {nproc} cores...", flush=True)
    with Pool(nproc) as pool:
        results = pool.map(field_job, work)
    acc = [[[] for _ in ALPHAS] for _ in BETAS]; dfl = [[] for _ in BETAS]
    for bi, rf, Df, g, per_alpha in results:
        dfl[bi].append(Df)
        if rf == 0: gap[bi] = g
        for ai in range(len(ALPHAS)): acc[bi][ai].extend(per_alpha[ai])
    for bi in range(len(BETAS)):
        Dfs[bi] = np.mean(dfl[bi]); Dfs_sd[bi] = np.std(dfl[bi])
        for ai in range(len(ALPHAS)):
            eta[bi, ai] = np.mean(acc[bi][ai]); eta_sd[bi, ai] = np.std(acc[bi][ai])
        print(f"beta={BETAS[bi]:.1f}  D_f={Dfs[bi]:.2f}  gapslope={gap[bi]:.2f}  "
              f"alpha*={ALPHAS[np.argmax(eta[bi])]:.2f}", flush=True)

    # optimal alpha* per beta with a parabolic refinement around the argmax
    astar = np.zeros(len(BETAS))
    for bi in range(len(BETAS)):
        j = int(np.argmax(eta[bi]))
        if 0 < j < len(ALPHAS) - 1:
            y0, y1, y2 = eta[bi, j-1], eta[bi, j], eta[bi, j+1]
            denom = (y0 - 2*y1 + y2)
            off = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            off = np.clip(off, -1, 1)
            astar[bi] = ALPHAS[j] + off * (ALPHAS[1] - ALPHAS[0])
        else:
            astar[bi] = ALPHAS[j]

    np.savez(os.path.join(HERE, "results.npz"), betas=BETAS, alphas=ALPHAS, eta=eta, eta_sd=eta_sd,
             Dfs=Dfs, Dfs_sd=Dfs_sd, gapslope=gap, astar=astar,
             params=dict(L=L, M=M, RV=RV, HOPS=HOPS, RF=RF, KS=KS, SEED0=SEED0))

    # (a) eta(alpha) curves
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(BETAS)))
    for bi, beta in enumerate(BETAS):
        e = eta[bi] / eta[bi].max()
        ax.plot(ALPHAS, e, "-o", color=cmap[bi], ms=4, label=f"β={beta:.1f} (D$_f$={Dfs[bi]:.2f})")
        ax.axvline(astar[bi], color=cmap[bi], ls=":", lw=1, alpha=0.6)
    ax.set_xlabel(r"search exponent $\alpha$ (max caliber)")
    ax.set_ylabel(r"foraging efficiency $\eta(\alpha)$ (normalised)")
    ax.set_title("Optimal search exponent shifts with field clustering")
    ax.legend(fontsize=8, title="resource field")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_eta_vs_alpha.pdf"))
    fig.savefig(os.path.join(HERE, "fig_eta_vs_alpha.png"), dpi=130)

    # (b) alpha* vs field exponent (test of matching)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(BETAS, astar, "o-", color="#1250f0", ms=6, label=r"optimal search $\alpha^*(\beta)$")
    lo = min(BETAS.min(), astar.min()); hi = max(BETAS.max(), astar.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label=r"match $\alpha^*=\beta$")
    ax.set_xlabel(r"resource-field exponent $\beta$ (max entropy)")
    ax.set_ylabel(r"optimal search exponent $\alpha^*$ (max caliber)")
    ax.set_title("Does the search exponent track the field?")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_alphastar_vs_beta.pdf"))
    fig.savefig(os.path.join(HERE, "fig_alphastar_vs_beta.png"), dpi=130)

    # correlation of alpha* with beta and with D_f
    cb = np.corrcoef(BETAS, astar)[0, 1]
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write("# maxent–caliber results\n\n")
        f.write(f"- params: L={L}, M={M}, rv={RV}, hops={HOPS}, fields/beta={RF}, searchers={KS}\n")
        f.write(f"- corr(alpha*, beta) = {cb:.2f}\n\n")
        f.write("| beta | D_f | gap-slope | alpha* |\n|---|---|---|---|\n")
        for bi, beta in enumerate(BETAS):
            f.write(f"| {beta:.1f} | {Dfs[bi]:.2f} | {gap[bi]:.2f} | {astar[bi]:.2f} |\n")
    print(f"\ncorr(alpha*, beta) = {cb:.2f}")
    print("alpha*(beta):", dict(zip(BETAS.tolist(), np.round(astar, 2).tolist())))
    print("wrote figs + results to", HERE)

if __name__ == "__main__":
    main()
