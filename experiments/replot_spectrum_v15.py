#!/usr/bin/env python3
"""CPU replot of Fig 7 (fig_powerspectrum) from the cached spectrum.npz -- figure polish only, no re-sim.
Single panel: the activity spectrum S_A(f). The 1/f and 1/f^2 guide lines are anchored at the roll-off
corner f_c (at the curve's value there) so the measured descent sits BETWEEN the two slopes; the corner is
marked with the autocorrelation time tau_ac (f_c = 1/(2 pi tau_ac)). The separate autocorrelation panel was
dropped -- it was redundant with the corner, which tau_ac already sets."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
d = np.load(os.path.join(HERE, "sweeps", "spectrum.npz"))
freqs, psd_m = d["freqs"], d["psd"]
beta_spec = float(d["beta_spec"]); tau_ac = float(d["tau_ac"])
fc = 1 / (2 * np.pi * tau_ac) if tau_ac > 0 else np.nan

fig, ax = plt.subplots(figsize=(6.3, 4.3))
m = freqs > 0
fm, pm = freqs[m], psd_m[m]
ax.loglog(fm, pm, color="#1f4e79", lw=1.7)
if fc == fc:
    ax.axvline(fc, color="crimson", ls=":", lw=1.4,
               label=f"corner $f_c\\approx{fc:.3f}$  ($\\tau_{{ac}}\\approx{tau_ac:.0f}$ ticks)")
    i_fc = int(np.argmin(np.abs(fm - fc))); y0 = pm[i_fc]       # anchor guides at the roll-off corner
    g = np.array([fc, fm[-1]])
    ax.loglog(g, y0 * (g / fc) ** -1, "k--", lw=0.9, alpha=0.6, label="$1/f$")
    ax.loglog(g, y0 * (g / fc) ** -2, "k:", lw=0.9, alpha=0.6, label="$1/f^2$")
ax.set_xlabel("frequency $f$"); ax.set_ylabel("$S_A(f)$")
ax.set_title(f"activity spectrum $A(t)$ ($\\beta_{{\\rm spec}}\\approx{beta_spec:.2f}$)")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig_powerspectrum.pdf"))
print("wrote fig_powerspectrum.pdf (single panel) | f_c=%.3f tau_ac=%.0f beta_spec=%.2f" % (fc, tau_ac, beta_spec))
