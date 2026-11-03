import os
#!/usr/bin/env python3
"""
RQ2 Fig 7 (spectrum) on the CORRECTED canonical engine (Code/mobile_cascade, short-step) — a fresh mobile
driver because the old run_spectra used the retired batched_swarm. Measures the activity spectrum S_A(f) of
A(t)=active_count/N at the swarm's operating point (replenished sources -> stationary A(t)), from scratch.
Reports whether it is Lorentzian (flat plateau + corner, finite correlation time) vs 1/f. Welch PSD + DFA
exponent + autocorrelation time; writes fig_powerspectrum.pdf into v15/figures and spectrum.npz.
Config = the gate's measured operating point: rho_r=3, theta(threshold)=0.20, phi(coverage)=0.05, alpha=1.5,
near-CSR, capacity=40 (depletes+respawns -> stationary). NOTATION: code theta=threshold, phi=coverage.
"""
import sys, time, os
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
sys.path.insert(0, _P.CODE)
import numpy as np, torch
import scipy.signal as ss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mobile_cascade import MobileCascade, MobileCascadeConfig

FIGDIR = _P.FIG
DATADIR = os.path.join(_P.DATA, "sweeps")

def dfa(x, scales):                       # detrended fluctuation analysis -> exponent alpha_DFA
    y = np.cumsum(x - x.mean()); F = []
    for s in scales:
        ns = len(y) // s
        if ns < 2: continue
        seg = y[:ns*s].reshape(ns, s); t = np.arange(s)
        rms = [np.sqrt(((row - np.polyval(np.polyfit(t, row, 1), t))**2).mean()) for row in seg]
        F.append((s, np.mean(rms)))
    F = np.array(F)
    return float(np.polyfit(np.log(F[:, 0]), np.log(F[:, 1]), 1)[0])

def main(B=192, N=150, T=16384, burn=2000, device="cuda", seed=0):
    cfg = MobileCascadeConfig(B=B, N=N, K=30, L=800.0, rho_r=3.0, phi=0.05, theta=0.20, step=20.0,
                              levy_alpha=1.5, exploit_step=3.0, consumption=1.0, capacity=40.0,
                              clump_count=None, converge=True, device=device, seed=seed)
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = MobileCascade(cfg).run(T)
    A = (out["active_count"].float() / N).cpu().numpy()[:, burn:]     # (B, T-burn)
    zc = float(out["z_local"][:, burn:].float().mean().item())
    nper = min(2048, A.shape[1])
    psd = np.array([ss.welch(A[b] - A[b].mean(), fs=1.0, nperseg=nper)[1] for b in range(A.shape[0])])
    freqs = ss.welch(A[0] - A[0].mean(), fs=1.0, nperseg=nper)[0]
    psd_m = psd.mean(0)
    scales = np.unique(np.round(np.logspace(np.log10(4), np.log10(A.shape[1]//4), 14)).astype(int))
    a_dfa = float(np.mean([dfa(A[b], scales) for b in range(0, A.shape[0], max(1, B//24))]))
    beta_spec = 2*a_dfa - 1
    ac = np.mean([np.correlate(A[b]-A[b].mean(), A[b]-A[b].mean(), "full")[A.shape[1]-1:] /
                  ((A[b]-A[b].mean())**2).sum() for b in range(0, A.shape[0], max(1, B//24))], axis=0)
    tau_ac = float(np.argmax(ac < 1/np.e)) if np.any(ac < 1/np.e) else np.nan
    os.makedirs(DATADIR, exist_ok=True)
    np.savez(os.path.join(DATADIR, "spectrum.npz"), freqs=freqs, psd=psd_m, ac=ac[:80],
             a_dfa=a_dfa, beta_spec=beta_spec, tau_ac=tau_ac, zc=zc, A_mean=float(A.mean()))
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    m = freqs > 0
    ax[0].loglog(freqs[m], psd_m[m], color="#1f4e79", lw=1.4)
    fc = 1/(2*np.pi*tau_ac) if tau_ac == tau_ac and tau_ac > 0 else np.nan
    if fc == fc: ax[0].axvline(fc, color="crimson", ls=":", lw=1.2, label=f"$f_c\\approx${fc:.3f}")
    g = np.array([freqs[m][1], freqs[m][-1]])
    ax[0].loglog(g, psd_m[m][1]*(g/g[0])**-1, "k--", lw=0.8, alpha=0.6, label="$1/f$")
    ax[0].loglog(g, psd_m[m][1]*(g/g[0])**-2, "k:", lw=0.8, alpha=0.6, label="$1/f^2$")
    ax[0].set_xlabel("frequency $f$"); ax[0].set_ylabel("$S_A(f)$")
    ax[0].set_title(f"activity spectrum ($\\beta_{{spec}}\\approx${beta_spec:.2f})"); ax[0].legend(fontsize=8)
    ax[1].semilogy(np.arange(len(ac[:80])), np.clip(ac[:80], 1e-3, None), color="#1f4e79", lw=1.4)
    ax[1].axhline(1/np.e, color="0.5", ls=":", lw=1); ax[1].set_xlabel("lag (ticks)")
    ax[1].set_ylabel("autocorr $A(t)$"); ax[1].set_title(f"$\\tau_{{ac}}\\approx${tau_ac:.0f} ticks")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_powerspectrum.pdf"))
    vram = torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else 0.0
    print(f"spectrum done: {time.time()-t0:.0f}s, VRAM {vram:.2f}GB, z_local={zc:.2f}, A_mean={A.mean():.3f}")
    print(f"  beta_spec(DFA)={beta_spec:.2f}, tau_ac={tau_ac:.0f} ticks, f_c={fc:.3f}")
    print(f"  => Lorentzian/not-1/f if beta_spec<1 and a flat plateau below f_c (finite correlation time)")

if __name__ == "__main__":
    main()
