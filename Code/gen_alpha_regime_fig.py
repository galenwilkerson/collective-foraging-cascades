#!/usr/bin/env python3
"""Figure: the CORRECTED Q4 after the frozen-agents motion fix.

Pre-fix, recruitment looked x2.43 cheaper on movement at alpha=1 -- but that was an artifact of
recruited agents freezing. With the fix (directed Levy strides, same length law as exploration)
recruitment is movement-NEUTRAL. What survives is a modest job-completion RATE advantage, conditional
on the exploration regime (low alpha) and resource sparsity, paid for with a ~5x activation-message
premium.

Pools q4_recheck.csv + q4_confirm.csv (two independent seeds). Ratio = baseline / recruit, paired
within context. -> theory_paper/fig_alpha_regime.pdf
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

D = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/Code/sweeps/"
rng = np.random.default_rng(0)

def load(f):
    d = pd.read_csv(D + f).replace([np.inf, -np.inf], np.nan)
    k = ["cell_alpha", "cell_cover", "cell_spc", "cell_phi", "ctx"]
    p = d.pivot_table(index=k, columns="cond",
                      values=["steps_per_target", "ticks_per_target",
                              "activation_per_target", "censored"]).dropna()
    ok = (p[("censored", "recruit")].values == 0) & (p[("censored", "baseline")].values == 0)
    return pd.DataFrame({
        "a": p.index.get_level_values("cell_alpha"),
        "phi": p.index.get_level_values("cell_phi"),
        "move": p[("steps_per_target", "baseline")] / p[("steps_per_target", "recruit")],
        "rate": p[("ticks_per_target", "baseline")] / p[("ticks_per_target", "recruit")],
        "acts": p[("activation_per_target", "baseline")] / p[("activation_per_target", "recruit")],
    })[ok]

df = pd.concat([load("q4_recheck.csv"), load("q4_confirm.csv")], ignore_index=True)

def ci(v):
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return (np.nan,) * 3
    b = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
    return np.percentile(b, [2.5, 50, 97.5])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))

# Panel A: the three cost axes vs alpha
alphas = np.sort(df.a.unique())
for key, col, lab, mk in [("move", "#1f77b4", "movement (baseline/recruit)", "o"),
                          ("rate", "#2ca02c", "completion rate (baseline/recruit)", "s"),
                          ("acts", "#d62728", "recruitment messages (baseline/recruit)", "^")]:
    m, lo, hi = [], [], []
    for a in alphas:
        c = ci(df[df.a == a][key].values); lo.append(c[0]); m.append(c[1]); hi.append(c[2])
    m = np.array(m)
    ax[0].errorbar(alphas, m, yerr=[m - np.array(lo), np.array(hi) - m], fmt=mk + "-",
                   color=col, capsize=3, ms=6, lw=1.6, label=lab)
ax[0].axhline(1.0, color="k", lw=1, ls="--")
# (pre-fix "artifact" annotation intentionally omitted; the bug is disclosed in the caption/footnote)
ax[0].set_xlabel("Lévy exponent $\\alpha$"); ax[0].set_ylabel("ratio")
ax[0].legend(fontsize=10, loc="upper right"); ax[0].set_ylim(0.0, 1.9)
ax[0].set_title("(a)", loc="left", fontweight="bold", fontsize=13)

# Panel B: rate advantage vs resource abundance phi
phis = np.sort(df.phi.unique())
m, lo, hi = [], [], []
for f in phis:
    c = ci(df[df.phi == f]["rate"].values); lo.append(c[0]); m.append(c[1]); hi.append(c[2])
m = np.array(m)
ax[1].errorbar(phis, m, yerr=[m - np.array(lo), np.array(hi) - m], fmt="s-", color="#2ca02c",
               capsize=3, ms=6, lw=1.6)
ax[1].axhline(1.0, color="k", lw=1, ls="--")
ax[1].set_xscale("log"); ax[1].set_xlabel(r"resource abundance $\varphi$ (sparse $\to$ abundant)")
ax[1].set_ylabel("completion-rate ratio (baseline/recruit)")
ax[1].text(phis[0], 1.02, "recruit faster", fontsize=11, color="#2ca02c")
ax[1].set_title("(b)", loc="left", fontweight="bold", fontsize=13)

# (suptitle omitted; the LaTeX caption carries the interpretation)
fig.tight_layout()
out = "/home/galen/Dropbox_2025/Dropbox/projects/flocking_LTM/theory_paper_v3/figures/fig_alpha_regime.pdf"
fig.savefig(out); print("wrote", out)

# print the numbers going into the caption
print("\nmovement ratio by alpha:", {float(a): round(ci(df[df.a==a]['move'].values)[1],2) for a in alphas})
print("rate ratio by alpha:    ", {float(a): round(ci(df[df.a==a]['rate'].values)[1],2) for a in alphas})
print("rate ratio sparse->abund:", [round(ci(df[df.phi==f]['rate'].values)[1],2) for f in phis])
