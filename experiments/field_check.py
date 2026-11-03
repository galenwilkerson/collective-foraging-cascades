#!/usr/bin/env python3
"""
FIELD FAITHFULNESS CHECK (CPU, field-only, no swarm). Is the Gaussian-clump (Thomas/Neyman-Scott) field
the swarm uses a faithful instance of the Gibbs point process the paper describes? Compare, at MATCHED
density (N=260 on torus [0,1]^2), the clustering STRENGTH (beta_eff, the paper's second-virial measure)
and STRUCTURE (pair correlation g(r)) of:
  POISSON (CSR = basic max-ent),  THOMAS/clump (several offspring scales),  GIBBS (attractive a<0).
Samplers/metrics copied verbatim from gen_beta_spectrum_fig.py and gen_beta_sparsity_fig.py.
Verdict: if a Thomas config matches a Gibbs config on BOTH beta_eff AND g(r), clumps are justified;
if it matches strength but not the g(r) shape, the swarm sweep should use the real Gibbs field.
"""
import numpy as np
rng = np.random.default_rng(0)
N = 260
def minimg(d): return d - np.round(d)

# --- Gibbs sampler (from gen_beta_spectrum_fig.py) ---
SIG, RMIN = 0.055, 0.012
def energy_point(pts, i, p, a):
    d = minimg(pts - p); r2 = (d**2).sum(1); k = np.exp(-r2/SIG**2); k[i] = 0.0
    return a * k.sum()
def too_close(pts, i, p):
    d = minimg(pts - p); r2 = (d**2).sum(1); r2[i] = 1.0
    return (r2 < RMIN**2).any()
def gibbs(a, sweeps=40):
    pts = rng.random((N, 2))
    for _ in range(sweeps):
        for i in rng.permutation(N):
            pn = rng.random(2)
            if too_close(pts, i, pn): continue
            dE = energy_point(pts, i, pn, a) - energy_point(pts, i, pts[i], a)
            if dE <= 0 or rng.random() < np.exp(-dE): pts[i] = pn
    return pts

# --- Thomas/clump field (from gen_beta_sparsity_fig.py) ---
def thomas(n_par, mu, sigc):
    par = rng.random((n_par, 2))
    off = par[:, None, :] + sigc * rng.standard_normal((n_par, mu, 2))
    return (off.reshape(-1, 2)) % 1.0

def poisson(): return rng.random((N, 2))

# --- metrics ---
R0 = 0.025
def close_pairs(pts, r0):
    d = minimg(pts[:, None, :] - pts[None, :, :]); r2 = (d**2).sum(2)
    iu = np.triu_indices(len(pts), 1); return int((r2[iu] < r0**2).sum())
def beta_eff(pts):
    n = len(pts); c = close_pairs(pts, R0); c_csr = n*(n-1)/2.0*(np.pi*R0**2)
    return np.log(max(c,1)/c_csr)
RG = np.array([0.010, 0.020, 0.040, 0.080, 0.150])
def gofr_at(pts):
    d = pts[:, None, :] - pts[None, :, :]; r = np.sqrt((minimg(d)**2).sum(2))
    r = r[np.triu_indices(len(pts), 1)]; rho = len(pts)/1.0; out = []
    for rr in RG:
        lo, hi = rr*0.8, rr*1.2
        cnt = ((r >= lo) & (r < hi)).sum()
        ideal = rho*len(pts)*np.pi*(hi**2 - lo**2)/2.0
        out.append(cnt/max(ideal, 1e-9))
    return np.array(out)

def stats(name, gen, reps=8):
    bs, gs = [], []
    for _ in range(reps):
        p = gen(); bs.append(beta_eff(p)); gs.append(gofr_at(p))
    bs = np.array(bs); gs = np.array(gs)
    print(f"  {name:22} beta_eff={bs.mean():5.2f}+-{bs.std():.2f}   g(r)@{list(RG)} = {np.round(gs.mean(0),2)}")

print(f"g(r) sampled at radii {list(RG)}  (CSR=1; clustered>1 at small r)")
stats("POISSON (max-ent/CSR)", poisson)
stats("THOMAS n13 mu20 s.015", lambda: thomas(13, 20, 0.015))
stats("THOMAS n13 mu20 s.030", lambda: thomas(13, 20, 0.030))
stats("THOMAS n13 mu20 s.050", lambda: thomas(13, 20, 0.050))
stats("GIBBS a=-0.6", lambda: gibbs(-0.6))
stats("GIBBS a=-0.95", lambda: gibbs(-0.95))
print("\n[Match = a THOMAS row whose beta_eff AND g(r) profile both line up with a GIBBS row.]")
