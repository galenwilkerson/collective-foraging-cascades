#!/usr/bin/env python3
"""
RQ4 Q4a/b: WHERE does recruitment pay, and what does it cost? CORRECTED-COORDINATE SOBOL, fixed-horizon,
on the CORRECTED engine -> v15.

Efficient space-filling Sobol over the joint parameter space (the paper's method, de-confounded), NOT a
dense grid. The one fix vs the old sweep: the two clustering columns are sampled in CORRECTED coordinates
-- cover in (0.005, 0.8) (<1 by construction) and src_per_clump in (1, 25) -- then inverted to the raw
(clump_count, patchiness), so "clumped" is never physically uniform (cover>1). theta is NOT sampled: each
Sobol point is run as a PAIR, recruit (theta_lo) vs no-recruit baseline (theta_base), same field+seed, so
the harvest/cost ratios are paired; every other axis (phi, alpha, rho_r, N, K, s_rel, ars_ratio,
deplete_ticks, dist_cat) varies with the point and its effect is recovered downstream as a marginal.

Measurement = fixed-horizon tail-window ledger (standing method, no completion/censoring): per replica,
harvest U, movement D, activation msgs Ma. Saved per point: all axis values + pooled sums (over B) + raw
per-replica U (for downstream bootstrap). Control validated by baseline cascade size ~0 vs recruit.

Downstream (separate aggregator): bin -> (cover x src_per_clump) harvest-ratio phase diagram; per-axis
binned marginals (harvest ratio & movement cost vs cover, src_per_clump, phi, alpha); per-axis importance
(the "strongest driver" question) -- all from this ONE sample. No target number assumed.
"""
import os, sys, time, argparse
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
import numpy as np, pandas as pd, torch
sys.path.insert(0, _P.CODE)
from mobile_cascade import MobileCascade, MobileCascadeConfig
from phase_space_sweep import AXES, unit_to_params
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
# Simulation data lives on /mnt/newssd, never in the (Dropbox-backed) repo (standing rule).
DATADIR = os.path.join(_P.DATA, "q4_sobol_v15")
IX_PATCH = [a[0] for a in AXES].index("patchiness")     # Sobol column reinterpreted as cover
IX_CLUMP = [a[0] for a in AXES].index("clump_count")    # Sobol column reinterpreted as src_per_clump
IX_RHO = [a[0] for a in AXES].index("rho_r")            # Sobol column reinterpreted as in-window z
COVER_LO, COVER_HI = 0.005, 0.8                          # cover < 1 by construction (de-confounded)
SPC_LO, SPC_HI = 1.0, 25.0
Z_LO, Z_HI = 4.5, 10.0        # sample z (=pi*p*rho_r) IN-WINDOW [z_c^RGG .. 2/theta_lo]: Q4 is the field
                              # question AT the swarm's operating regime (Q1 owns the z/window dependence).


@torch.no_grad()
def ledger(theta, P, args, baseline):
    spread = (0.0 if baseline else P["s_rel"]) * theta
    cfg = MobileCascadeConfig(
        B=args.B, N=P["N"], K=P["K"], L=args.L, rho_r=P["rho_r"], p=P["p"], phi=P["phi"], theta=theta,
        theta_dist=("uniform" if P["dist_cat"] else "gaussian") if spread > 0 else "const", theta_std=spread,
        step=args.step, exploit_step=P["ars_ratio"] * args.step, pull=P["pull"], levy_alpha=P["levy_alpha"],
        clump_count=P["clump_count"], clump_radius=P["patchiness"] * args.L,
        capacity=P["deplete_ticks"] * args.consumption, consumption=args.consumption,
        converge=True, device=args.device, seed=args.seed)
    o = MobileCascade(cfg).run(args.T)
    t0 = int((1.0 - args.tail) * args.T); Tw = args.T - t0; den = args.consumption * P["N"] * Tw
    U = o["consumed"][:, t0:].sum(1).cpu().numpy() / den
    D = o["distance"][:, t0:].sum(1).cpu().numpy() / (P["N"] * Tw * args.step)
    Ma = o["msgs_activation"][:, t0:].sum(1).cpu().numpy() / (P["N"] * Tw)
    cs = float(o["cascade_size"].mean().item())
    act = float((1.0 - o["exploring_fraction"][:, t0:].mean()).item())
    return U, D, Ma, cs, act


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=512, help="Sobol points (power of 2)")
    ap.add_argument("--offset", type=int, default=0, help="skip this many Sobol points (for Anvil array sharding)")
    ap.add_argument("--theta_lo", type=float, default=0.20); ap.add_argument("--theta_base", type=float, default=0.99)
    ap.add_argument("--B", type=int, default=32); ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--tail", type=float, default=0.5); ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--step", type=float, default=20.0); ap.add_argument("--consumption", type=float, default=1.0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="q4_sobol_v15")
    args = ap.parse_args()
    os.makedirs(DATADIR, exist_ok=True)
    total = args.offset + args.M
    sob = qmc.Sobol(d=len(AXES), scramble=True, seed=args.seed).random(total)[args.offset:]
    if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); rows, raw = [], {}
    for j, u in enumerate(sob):
        P = unit_to_params(u)
        P["rho_r"] = (Z_LO * (Z_HI / Z_LO) ** u[IX_RHO]) / (np.pi * P["p"])   # z in-window for ALL points
        cover = COVER_LO * (COVER_HI / COVER_LO) ** u[IX_PATCH]
        spc = SPC_LO * (SPC_HI / SPC_LO) ** u[IX_CLUMP]
        nclump = max(1, int(round(P["K"] / spc)))
        patch = float(np.sqrt(cover / (nclump * np.pi)))
        if patch > 0.7:                              # cover not realizable inside the arena -> skip
            continue
        P["clump_count"], P["patchiness"] = nclump, patch
        Ur, Dr, Mar, csr, actr = ledger(args.theta_lo, P, args, baseline=False)
        Ub, Db, Mab, csb, actb = ledger(args.theta_base, P, args, baseline=True)
        pj = args.offset + j
        raw[f"p{pj}__Urec"] = Ur.astype(np.float32); raw[f"p{pj}__Ubase"] = Ub.astype(np.float32)
        rows.append(dict(point=pj, cover=cover, actual_cover=float(nclump * np.pi * patch ** 2),
            src_per_clump=float(P["K"] / nclump), phi=P["phi"], alpha=P["levy_alpha"],
            rho_r=P["rho_r"], p=P["p"], z=float(np.pi * P["p"] * P["rho_r"]), N=int(P["N"]), K=int(P["K"]),
            s_rel=P["s_rel"], ars_ratio=P["ars_ratio"], deplete_ticks=P["deplete_ticks"], dist_cat=P["dist_cat"],
            sumU_rec=float(Ur.sum()), sumU_base=float(Ub.sum()), sumD_rec=float(Dr.sum()),
            sumD_base=float(Db.sum()), sumMa_rec=float(Mar.sum()), cascade_rec=csr, cascade_base=csb,
            active_rec=actr, active_base=actb))
        if len(rows) % 32 == 0:
            pd.DataFrame(rows).to_csv(os.path.join(DATADIR, f"{args.tag}_off{args.offset}.csv"), index=False)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(DATADIR, f"{args.tag}_off{args.offset}.csv"), index=False)
    np.savez(os.path.join(DATADIR, f"{args.tag}_off{args.offset}_raw.npz"), **raw)
    vram = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0
    print(f"\n=== Q4 Sobol smoke: {time.time()-t0:.0f}s, VRAM {vram:.2f}GB, {len(df)} points (M={args.M}) ===")
    print(f"CONTROL baseline cascade mean={df.cascade_base.mean():.3f} max={df.cascade_base.max():.2f} "
          f"(recruit {df.cascade_rec.mean():.1f}); ~0 => clean no-recruit control")
    for lab, col in [("cover(sparse->diffuse)", "cover"), ("src_per_clump(clustering)", "src_per_clump"),
                     ("alpha", "alpha")]:
        q = pd.qcut(df[col], min(3, df[col].nunique()), duplicates="drop")
        gr = df.groupby(q, observed=True).apply(lambda g: g.sumU_rec.sum() / max(g.sumU_base.sum(), 1e-12))
        print(f"  harvest ratio by {lab} tercile: " + ", ".join(f"{v:.2f}" for v in gr.values))


if __name__ == "__main__":
    main()
