#!/usr/bin/env python3
"""RQ5 SI fig_scalefree on the CORRECTED engine -> v15. Reads scale_free_field_v15.npz and plots, vs the
realized resource exponent beta_res: (a) harvest ratio recruit/baseline (>1 = recruitment pays), (b) the
emergent active-cluster exponent tau_active (vs mean-field 3/2). Error bars = SEM over per-replica reps."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "sweeps", "scale_free_field_v15.npz"), allow_pickle=True)
summ = d["summary"]                       # cols: input_beta, beta_res, harvest_ratio, tau_active
bres = summ[:, 1]; hr = summ[:, 2]; tau = summ[:, 3]
hr_sem = np.nanstd(d["harvest_ratio_reps"], axis=1) / np.sqrt(d["harvest_ratio_reps"].shape[1])
tau_sem = np.nanstd(d["tau_reps"], axis=1) / np.sqrt(np.sum(~np.isnan(d["tau_reps"]), axis=1).clip(1))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.7))
a1.axhline(1, ls=":", color="0.5", lw=1, label="parity")
a1.errorbar(bres, hr, yerr=hr_sem, marker="o", lw=1.8, color="#1f4e79", capsize=3)
a1.set_xlabel(r"realized resource exponent $\beta_{\rm res}$"); a1.set_ylabel("harvest ratio recruit/baseline")
a1.set_title("(a) recruitment pays across the scale-free regime", fontsize=11); a1.legend(fontsize=9)
a2.axhline(1.5, ls="--", color="#c0392b", lw=1.2, label="mean-field $3/2$")
a2.errorbar(bres, tau, yerr=tau_sem, marker="s", lw=1.8, color="#2e7d32", capsize=3)
a2.set_xlabel(r"realized resource exponent $\beta_{\rm res}$"); a2.set_ylabel(r"active-cluster exponent $\tau_{\rm active}$")
a2.set_title("(b) emergent exponent $\\approx$ const, below $3/2$", fontsize=11); a2.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "figures", "fig_scalefree.pdf"); fig.savefig(out)
print("wrote", out)
print("beta_res:", np.round(bres, 2), "| harvest ratio:", np.round(hr, 2), "| tau_active:", np.round(tau, 2))
