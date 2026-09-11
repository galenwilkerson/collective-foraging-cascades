#!/usr/bin/env python3
"""Regenerate the SOC-cycle time series WITH the node-state split (blocked/isolated), all replicas, for
the state-coloured phase portrait. Same cycling config as mobile_soc.npz (rho_r=3, phi=0.05, theta=0.15,
coupled). mobile_cascade already computes every state; we just save them.

Node states (partition all N agents at the cascade fixed point):
  consuming = active_count - cascade_size   (on-food seeds)
  recruited = cascade_size                  (active via cascade, off food)
  blocked   = blocked                       (inactive WITH an active neighbour, a/k<theta)
  exploring = isolated                      (inactive with NO active neighbour)

Data -> /mnt/newssd (per Galen's rule), symlinked into Code/sweeps. CPU (GPU is held by another instance).
"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mobile_cascade import MobileCascade, MobileCascadeConfig

OUTDIR = os.path.join(HERE, "sweeps")
LINKDIR = os.path.join(HERE, "sweeps")
NAME = "cycle_states.npz"

cfg = MobileCascadeConfig(B=128, N=100, K=30, rho_r=3.0, phi=0.05, theta=0.15, step=20.0, pull=0.1,
                          consumption=1.0, capacity=40.0, converge=True, seed=0,
                          levy_alpha=1.5, exploit_step=3.0)   # Levy branch carries the convergence coupling
sim = MobileCascade(cfg)
print(f"running coupled cycle: B={cfg.B} N={cfg.N} K={cfg.K} T=2000 theta={cfg.theta} on {sim.device}", flush=True)
out = sim.run(2000)
o = {k: v.detach().cpu().numpy().astype("float32") for k, v in out.items()}
N = sim.N

keys = [k for k in ["active_count", "cascade_size", "blocked", "isolated",
                    "z_local", "z_global", "resource", "sigma", "exploring_fraction"] if k in o]
save = {k: o[k] for k in keys}
save["N"] = np.float32(N)
save["theta"] = np.float32(cfg.theta)
print("saved keys:", keys, "| shapes:", {k: o[k].shape for k in keys[:3]}, flush=True)

os.makedirs(OUTDIR, exist_ok=True)
path = os.path.join(OUTDIR, NAME)
np.savez(path, **save)
print("wrote", path, flush=True)
print(f"wrote {path}\nsymlinked -> {link}", flush=True)
