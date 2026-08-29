#!/usr/bin/env python3
"""Generate publication PDF figures for main.tex from the sweep data in ../Code/sweeps."""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RW_C = "#3b6ea5"   # random walk (blue)
LV_C = "#e07b39"   # Levy (orange)

SW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Code", "sweeps")
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "figure.dpi": 150})


def fig_phase_diagram():
    RR = np.load(os.path.join(SW, "resource_regimes.npz"))
    patch, phis, H = RR["patchiness"], RR["phis"], RR["harvest"]
    ext = [patch.min(), patch.max(), phis.min(), phis.max()]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(H, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=0, vmax=2)
    # phase boundary where recruitment ties the baseline (ratio = 1)
    PP, FF = np.meshgrid(patch, phis)
    ax.contour(PP, FF, H, levels=[1.0], colors="k", linewidths=2, linestyles="--")
    ax.set_xlabel(r"patchiness  (clump radius / $L$: tight $\rightarrow$ diffuse)")
    ax.set_ylabel(r"abundance  $\phi$")
    cb = fig.colorbar(im, ax=ax); cb.set_label(r"$U_{\rm recruit}/U_{\rm base}$  (>1: recruitment pays)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_phasediagram.pdf")); print("wrote fig_phasediagram.pdf")


def fig_fss():
    F = np.load(os.path.join(SW, "fss_scaling.npz")); z = F["z"]; Ns = F["Ns"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for N in Ns:
        a1.plot(z, F[f"cascade_{N}"], "-o", ms=3, label=f"N={N}")
        a2.plot(z, F[f"chi_{N}"], "-o", ms=3, label=f"N={N}")
    a1.set_xlabel("mean degree $z$"); a1.set_ylabel("cascade fraction")
    a1.legend(fontsize=8, frameon=False)
    a2.set_xlabel("mean degree $z$"); a2.set_ylabel(r"$\chi = N\,\mathrm{Var}(\Psi)$")
    a2.legend(fontsize=8, frameon=False)
    a1.set_title("(a)", loc="left", fontweight="bold", fontsize=13)
    a2.set_title("(b)", loc="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_fss.pdf")); print("wrote fig_fss.pdf")


def _levy_mag(a, n, rng):
    """|alpha-stable| step-length magnitudes, exactly the Mantegna law the model draws (mobile_cascade._levy_length)."""
    num = (math.gamma(1 + a) * math.sin(math.pi * a / 2)) ** (1.0 / a)
    den = (math.gamma((1 + a) / 2) * a * 2 ** ((a - 1) / 2)) ** (1.0 / a)
    sigma = num / den
    u = rng.standard_normal(n) * sigma
    w = np.abs(rng.standard_normal(n))
    return np.abs(u / np.clip(w, 1e-12, None) ** (1.0 / a))


def _ccdf(x):
    x = np.sort(x)
    return x, 1.0 - np.arange(len(x)) / len(x)


def fig_local_clusters():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- (a) MOVEMENT: the step-length laws the agents draw from (median-scaled to compare shape) ----
    rng = np.random.default_rng(0)
    n = 400_000
    rw = rng.rayleigh(size=n)                                   # random-walk step = |2D Gaussian| ~ Rayleigh
    for lab, ls, col, sample in [
        ("random walk (Brownian)", "-", RW_C, rw),
        ("Lévy ($\\alpha=1.5$)", "-", LV_C, _levy_mag(1.5, n, rng)),
        ("Lévy ($\\alpha=1.0$)", "--", LV_C, _levy_mag(1.0, n, rng)),
    ]:
        s = sample / np.median(sample)                         # equal median -> pure tail-shape comparison
        x, c = _ccdf(s)
        a1.loglog(x, c, ls, color=col, lw=2, label=lab)
    a1.set_xlim(0.05, 3e3)
    a1.set_ylim(1e-5, 1.3)
    a1.set_xlabel(r"step length $\ell$  (median-scaled)")
    a1.set_ylabel(r"$P(L \geq \ell)$")
    a1.annotate("heavy tail:\nrare long relocations", xy=(130, 8e-4), xytext=(260, 6e-5),
                fontsize=8.5, color=LV_C, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=LV_C, lw=1, connectionstyle="arc3,rad=0.2"))
    a1.text(0.07, 8e-4, "Brownian:\nbounded (light tail)", fontsize=8.5, color=RW_C)
    a1.legend(fontsize=9, frameon=False, loc="upper right")

    # ---- (b) RECRUITMENT: emergent active-cluster sizes (same exponent across modes) ----
    for tag, col, lab in [("local_soc.npz", RW_C, "random walk"), ("local_soc_levy.npz", LV_C, "Lévy")]:
        L = np.load(os.path.join(SW, tag))
        th = float(L["thetas"][-1])                            # highest theta (cleanest power law)
        sz = L[f"sizes_{th:.2f}"]; sz = sz[sz >= 2]
        x, c = _ccdf(sz)
        a2.loglog(x, c, ".", ms=3, alpha=0.6, color=col, label=f"{lab} ($\\theta$={th:.2f})")
        m = (x >= 2) & (x <= x.max() * 0.5) & (c > 0)
        b = np.polyfit(np.log10(x[m]), np.log10(c[m]), 1)
        r2 = 1 - np.sum((np.log10(c[m]) - np.polyval(b, np.log10(x[m]))) ** 2) / \
            np.sum((np.log10(c[m]) - np.log10(c[m]).mean()) ** 2)
        a2.loglog(x[m], 10 ** b[1] * x[m] ** b[0], "-", color=col, lw=1.5,
                  label=f"  fit slope {b[0]:.2f}, $R^2$={r2:.2f}")
    a2.set_xlabel(r"active-cluster size $s$")
    a2.set_ylabel(r"$P(S \geq s)$")
    a2.legend(fontsize=8.5, frameon=False, markerscale=4, handletextpad=0.5)

    for ax, tag in [(a1, "(a)"), (a2, "(b)")]:
        ax.text(-0.13, 1.04, tag, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_localclusters.pdf")
    fig.savefig(out); print("wrote", out)


def fig_scale_free():
    F = np.load(os.path.join(SW, "scale_free_field.npz"))
    s = F["summary"]                      # cols: input_beta, beta_res, harvest_ratio, tau_active
    beta_res, ratio, tau = s[:, 1], s[:, 2], s[:, 3]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    a1.plot(beta_res, ratio, "o-", color="#2e8b57", ms=7, lw=2)
    a1.axhline(1, ls=":", color="k"); a1.set_ylim(0, max(3.2, ratio.max() * 1.15))
    a1.set_xlabel(r"resource cluster-size exponent $\beta_{\rm res}$")
    a1.set_ylabel("harvest ratio  recruit / baseline")
    a2.plot(beta_res, tau, "s-", color="#8b2d8b", ms=7, lw=2)
    a2.set_xlabel(r"resource cluster-size exponent $\beta_{\rm res}$")
    a2.set_ylabel(r"emergent active-cluster exponent $\tau_{\rm active}$")
    a1.set_title("(a)", loc="left", fontweight="bold", fontsize=13)
    a2.set_title("(b)", loc="left", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_scalefree.pdf")); print("wrote fig_scalefree.pdf")


if __name__ == "__main__":
    fig_phase_diagram(); fig_fss(); fig_local_clusters(); fig_scale_free()
