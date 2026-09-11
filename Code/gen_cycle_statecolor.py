#!/usr/bin/env python3
"""State-coloured empirical phase portrait (variant C2): the (z_local, active-fraction) plane, its
background coloured by NODE-STATE COMPOSITION among engaged agents -- recruit (blue), consume (green),
blocked (red); explore is the neutral/rest background -- with mean-flow streamlines on top. Reads
Code/sweeps/cycle_states.npz (128 replicas, from gen_cycle_data.py). CPU, seconds.
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "sweeps", "cycle_states.npz"))
N = float(d["N"]); THETA = float(d["theta"]); ZC_RGG = 4.51; Z_BLOCK = 2.0 / THETA
BURN, W = 200, 45
FIGS = os.path.normpath(os.path.join(HERE, "..", "theory_paper_v6", "figures"))
PNG = os.environ.get("SCRATCH_PNG_DIR", "/tmp")
plt.rcParams.update({"font.size": 16})

def sm(x): return np.convolve(x, np.ones(W) / W, "valid")

ac, cs, bl, iso, zl = d["active_count"], d["cascade_size"], d["blocked"], d["isolated"], d["z_local"]
rec, con, blk, act = cs / N, (ac - cs) / N, bl / N, ac / N          # fractions per (replica,tick)
exp = iso / N                                                       # explore = isolated (inactive, no active neighbour)

Zs, As, Rc, Cn, Bk, Ex, DZ, DA = [], [], [], [], [], [], [], []
for r in range(zl.shape[0]):
    z, a = sm(zl[r, BURN:]), sm(act[r, BURN:])
    rr, cc, bb, ee = sm(rec[r, BURN:]), sm(con[r, BURN:]), sm(blk[r, BURN:]), sm(exp[r, BURN:])
    Zs.append(z[:-1]); As.append(a[:-1]); Rc.append(rr[:-1]); Cn.append(cc[:-1]); Bk.append(bb[:-1]); Ex.append(ee[:-1])
    DZ.append(np.diff(z)); DA.append(np.diff(a))
Z, A, R_, C_, B_, E_ = map(np.concatenate, (Zs, As, Rc, Cn, Bk, Ex)); DZ, DA = map(np.concatenate, (DZ, DA))
XMAX = float(np.percentile(Z, 99))

fig, ax = plt.subplots(figsize=(6.8, 5.2))
ax.set_rasterization_zorder(6)   # flatten the busy layers (streamlines, patches, contours, nullclines)
#                                  into one embedded image so the PDF prints easily; text/star/legend stay vector

# --- background coloured by state composition (R=blocked, G=consume, B=recruit; among engaged) ---
nx, ny = 46, 38
xe, ye = np.linspace(0, XMAX, nx + 1), np.linspace(0, 1, ny + 1)
ix = np.clip(np.searchsorted(xe, Z) - 1, 0, nx - 1); iy = np.clip(np.searchsorted(ye, A) - 1, 0, ny - 1)
sr = np.zeros((ny, nx)); sc = np.zeros((ny, nx)); sb = np.zeros((ny, nx)); se = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
np.add.at(sr, (iy, ix), R_); np.add.at(sc, (iy, ix), C_); np.add.at(sb, (iy, ix), B_); np.add.at(se, (iy, ix), E_); np.add.at(cnt, (iy, ix), 1)
mm = cnt > 5                                                     # ignore low-count (noisy) bins
mr = np.where(mm, sr / np.maximum(cnt, 1), 0); mc = np.where(mm, sc / np.maximum(cnt, 1), 0)
mb = np.where(mm, sb / np.maximum(cnt, 1), 0); me = np.where(mm, se / np.maximum(cnt, 1), 0)
MODE = "patches" if os.environ.get("PATCHES") else ("dom" if os.environ.get("DOM") else "blend")
if MODE == "patches":                                            # colored "rich"-region loops, one per state
    from scipy.ndimage import gaussian_filter as _gf
    xcg, ycg = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    for name, fld, col in [("explore", me, "#8f8f8f"), ("recruit", mr, "#1f4fd8"),
                           ("blocked", mb, "#c01515"), ("consume", mc, "#1a7a1a")]:
        f = _gf(np.where(mm, fld, 0.0), 1.5)                     # smoothed fraction-of-N field for this state (no specks)
        lev = 1.5 * float(fld[mm].mean()) if mm.any() else 0.0  # enrichment contour: >=1.5x this state's plane-average
        if lev <= 0:                                            # (one uniform rule, but prevalence-aware)
            continue
        ax.contourf(xcg, ycg, f, levels=[lev, f.max() + 1e-6], colors=[col], alpha=0.33, zorder=1)
        ax.contour(xcg, ycg, f, levels=[lev], colors=[col], linewidths=1.7, zorder=5)
        ax.plot([], [], "s", color=col, alpha=0.75, ms=9, label=name + "-rich")
else:
    if MODE == "dom":                                            # DOMINANT-state patches (no blending, no purple)
        cols = np.array([[0.13, 0.31, 0.85], [0.11, 0.55, 0.13], [0.80, 0.11, 0.11], [0.60, 0.60, 0.60]])
        dom = np.argmax(np.dstack([mr, mc, mb, me]), axis=2)     # 0=recruit 1=consume 2=blocked 3=explore
        rgb = cols[dom]
        Rch, Gch, Bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        alpha = np.clip((cnt / cnt.max()) ** 0.32 * 1.2, 0, 1) * mm
    else:                                                        # blended (composition) coloring, saturation-boosted
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        eng = mr + mc + mb + 1e-9
        comp = np.clip(np.dstack([mb / eng, mc / eng, mr / eng]), 0, 1)   # R=blocked G=consume B=recruit
        hsv = rgb_to_hsv(comp); hsv[..., 1] = np.clip(hsv[..., 1] * 1.9, 0, 1)  # boost saturation
        comp = hsv_to_rgb(hsv)
        Rch, Gch, Bch = comp[..., 0], comp[..., 1], comp[..., 2]
        dens = mr + mc + mb                                      # engaged fraction -> alpha (explore fades to white)
        dref = np.percentile(dens[mm], 95) if mm.any() else 1.0
        alpha = np.clip((dens / (dref + 1e-9)) ** 0.7, 0, 1) * (cnt / cnt.max()) ** 0.15 * mm
    img = np.clip(np.dstack([Rch, Gch, Bch, alpha]), 0, 1)
    ax.imshow(img, origin="lower", extent=[0, XMAX, 0, 1], aspect="auto", zorder=1, interpolation="bilinear")
    if MODE == "blend":                                         # consume never wins a patch -> outline where it peaks
        from scipy.ndimage import gaussian_filter as _gf
        hi = cnt > 20
        xcg, ycg = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        mc_s = _gf(np.where(hi, mc, 0.0), 0.8)
        lev = float(np.percentile(mc[hi], 92)) if hi.any() else 0.0
        if lev > 0:
            ax.contour(xcg, ycg, mc_s, levels=[lev], colors="#0a6a0a", linewidths=1.8, zorder=6)
            ax.plot([], [], "-", color="#0a6a0a", lw=1.8, label="consume-rich")

# --- mean-flow streamlines ---
nx2, ny2 = 22, 16
xe2, ye2 = np.linspace(0, XMAX, nx2 + 1), np.linspace(0, 1, ny2 + 1)
xc2, yc2 = 0.5 * (xe2[:-1] + xe2[1:]), 0.5 * (ye2[:-1] + ye2[1:])
U = np.zeros((ny2, nx2)); V = np.zeros((ny2, nx2)); C2 = np.zeros((ny2, nx2))
jx = np.clip(np.searchsorted(xe2, Z) - 1, 0, nx2 - 1); jy = np.clip(np.searchsorted(ye2, A) - 1, 0, ny2 - 1)
np.add.at(U, (jy, jx), DZ); np.add.at(V, (jy, jx), DA); np.add.at(C2, (jy, jx), 1)
good = C2 > 12
U = np.where(good, U / np.maximum(C2, 1), 0); V = np.where(good, V / np.maximum(C2, 1), 0)
spd = np.hypot(U, V); lw = 0.5 + 2.0 * spd / (spd.max() + 1e-9)
if MODE == "patches":                                          # cleaner, sparser flow so the colour reads
    _scol, _sden, _slw, _sarr, _sasz = "0.5", 0.7, 0.9, "-|>", 1.6
else:
    _scol, _sden, _slw, _sarr, _sasz = "0.12", 1.1, lw, "-|>", 1.0
try:
    ax.streamplot(np.array(xc2), np.array(yc2), U, V, color=_scol, linewidth=_slw, density=_sden,
                  arrowstyle=_sarr, arrowsize=_sasz, zorder=3)
except Exception as e:
    print("streamplot fallback:", e)

# --- nullclines (dz=0 cyan, da=0 magenta) + fixed point (stable focus = poised rest point) ---
from scipy.ndimage import gaussian_filter
occ_ok = C2 > 20
Uf, Vf = gaussian_filter(U, 0.8), gaussian_filter(V, 0.8)
Ygrid = np.repeat(np.array(yc2)[:, None], len(xc2), axis=1)   # the focus is the swirl centre (nullcline
sp2 = np.hypot(Uf, Vf); sp2 = np.where(occ_ok & (Ygrid > 0.20), sp2, np.inf)  # crossing), above the rest band
fj, fi = np.unravel_index(np.argmin(sp2), sp2.shape); fx, fy = xc2[fi], yc2[fj]
Um, Vm = np.where(occ_ok, Uf, np.nan), np.where(occ_ok, Vf, np.nan)
ax.contour(xc2, yc2, Um, levels=[0], colors="#00b0b0", linewidths=1.2, linestyles="--", zorder=4)
ax.contour(xc2, yc2, Vm, levels=[0], colors="#b000b0", linewidths=1.2, linestyles="--", zorder=4)
ax.scatter([fx], [fy], s=150, marker="*", color="k", zorder=8)
ax.text(fx + 0.7, fy + 0.03, "poised operating point\n(stable focus)", fontsize=9, va="bottom", zorder=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=1.0))
ax.plot([], [], "--", color="#00b0b0", label=r"$\dot z=0$ nullcline"); ax.plot([], [], "--", color="#b000b0", label=r"$\dot a=0$ nullcline")
ax.legend(loc=("upper left" if MODE == "patches" else "lower right"), fontsize=10.5, frameon=True, framealpha=0.85)

# --- boundaries, labels, legend ---
for xv, lab in [(ZC_RGG, r"$z_c^{\rm RGG}$"), (Z_BLOCK, r"$2/\varphi$")]:
    ax.axvline(xv, color="0.5", ls=":", lw=1.1, zorder=2)
    ax.text(xv, 1.02, lab, ha="center", va="bottom", fontsize=11, color="0.4")
if MODE != "patches":                                            # cycle-phase labels + swatch key (blended/dom only)
    for x0, y0, t, col in [(0.13 * XMAX, 0.11, "explore", "#6f6f6f"),
                           (0.30 * XMAX, 0.34, "recruit /\nignite", "#1f4fd8"),
                           (0.66 * XMAX, 0.66, "consume", "#1a7a1a"),
                           (0.74 * XMAX, 0.30, "blocked /\ndisperse", "#b01515")]:
        ax.text(x0, y0, t, ha="center", va="center", fontsize=10.5, fontstyle="italic", color=col, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7), zorder=7)
    for i, (t, col) in enumerate([("recruit", "#1f4fd8"), ("consume", "#1a7a1a"), ("blocked", "#b01515"), ("explore", "#9e9e9e")]):
        ax.text(0.015, 0.95 - i * 0.055, "■ " + t, transform=ax.transAxes, color=col, fontsize=10,
                weight="bold", va="top", zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

ax.set_xlim(0, XMAX); ax.set_ylim(0, 1)
ax.set_xlabel(r"connectivity  $z_{\rm local}$  (mean degree of active agents)")
ax.set_ylabel("active fraction  (recruited $+$ consuming)")
fig.tight_layout()
NAME = {"patches": "fig_cycle_flow_patches", "dom": "fig_cycle_flow_dom", "blend": "fig_cycle_flow"}[MODE]
fig.savefig(os.path.join(FIGS, NAME + ".pdf"), bbox_inches="tight", dpi=300)  # 300 dpi for the rasterized layer
fig.savefig(os.path.join(PNG, NAME + ".png"), dpi=130, bbox_inches="tight")
print("wrote %s.pdf | XMAX=%.0f theta=%.2f focus=(%.1f,%.2f)" % (NAME, XMAX, THETA, fx, fy))
