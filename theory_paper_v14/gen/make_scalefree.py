"""Fig 10 (SI, fig:scalefree): scale-free resource fields. LEFT harvest ratio, RIGHT emergent exponent
tau_active, both vs resource cluster-size exponent beta_res. Matched panel style: markers + line +
shaded 95% bootstrap-CI band. tau point = per-replica median (inside its CI; the pooled single-fit
sits outside the CI for 2/3 points, which is why a band on it would look broken)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NPZ = os.path.join(REPO, "Code", "sweeps", "scalefree_v14.npz")
OUT = os.path.join(REPO, "theory_paper_v14", "figures", "fig_scalefree.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = np.load(NPZ, allow_pickle=True)
s = d["summary"]; beta = s[:, 1]; ratio = s[:, 2]
hr = d["harvest_ratio_reps"]; tr = d["tau_reps"]
rng = np.random.default_rng(0)


def ci(reps):
    lo = np.empty(len(reps)); hi = np.empty(len(reps)); med = np.empty(len(reps))
    for i, r in enumerate(reps):
        r = np.asarray(r); r = r[np.isfinite(r)]
        bs = [rng.choice(r, len(r)).mean() for _ in range(2000)]
        lo[i], hi[i] = np.percentile(bs, [2.5, 97.5]); med[i] = np.median(r)
    return lo, hi, med


hlo, hhi, _ = ci(hr); tlo, thi, tau_med = ci(tr)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
# LEFT: harvest ratio -- markers + line + shaded CI band
a1.plot(beta, ratio, "o-", color="#2e8b57", ms=7, lw=2, label="harvest ratio")
a1.fill_between(beta, hlo, hhi, color="#2e8b57", alpha=0.20, lw=0, label="95% bootstrap CI")
a1.axhline(1, ls=":", color="k"); a1.set_ylim(0, max(6.8, hhi.max() * 1.12))
a1.set_xlabel(r"resource cluster-size exponent $\beta_{\rm res}$")
a1.set_ylabel("harvest ratio  recruit / baseline")
a1.legend(fontsize=8, frameon=False, loc="upper right")
# RIGHT: emergent exponent -- SAME style (markers + line + shaded CI band)
a2.plot(beta, tau_med, "s-", color="#8b2d8b", ms=7, lw=2, label=r"$\tau_{\rm active}$")
a2.fill_between(beta, tlo, thi, color="#8b2d8b", alpha=0.20, lw=0, label="95% bootstrap CI")
a2.set_xlabel(r"resource cluster-size exponent $\beta_{\rm res}$")
a2.set_ylabel(r"emergent active-cluster exponent $\tau_{\rm active}$")
a2.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT, dpi=200)
print("beta", np.round(beta, 2), "| tau_med", np.round(tau_med, 3),
      "| harvest", np.round(ratio, 2), "| tau CI", [f"[{l:.2f},{h:.2f}]" for l, h in zip(tlo, thi)])
print("wrote", OUT)
