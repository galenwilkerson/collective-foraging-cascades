import os
#!/usr/bin/env python3
"""
RQ1 SMOKE (GPU): the cascade window = inverted-U of cascade fraction vs mean degree z.
Static RGG + Watts LTM fixed point (engine-independent). Theory overlays: z_c^RGG=4.51 (percolation),
2/phi blocking (Watts). We run at explicit threshold theta=phi=0.15 so the 2/phi=13.3 line matches the data.
Measures GPU timing + VRAM at a representative batch, and confirms the window shape (percolation onset ~4.5,
peak ~6-7, blocking falloff toward 2/phi). Purpose: confirm RQ1 runs efficiently on GPU before the full run.
"""
import sys, time, math
import _paths as _P  # repo path resolution (Code on sys.path; FIG/DATA dirs)
sys.path.insert(0, _P.CODE)
import numpy as np, torch
from inverted_u import sweep   # static primitives: build_rgg + ltm_fixed_point + gcc_fraction

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PHI = 0.15                      # threshold; 2/phi = 13.3 (blocking boundary, theory)
N, B = 500, 512
zgrid = np.array([2., 3., 4., 4.51, 5., 6., 7., 9., 11., 13.3, 16., 20.])
print(f"device={DEV}  N={N} B={B}  theta=phi={PHI}  (z_c^RGG=4.51, 2/phi={2/PHI:.1f})")

if DEV == "cuda":
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
rows = sweep(zgrid, N=N, B=B, theta=PHI, p=1.0, seed_frac=0.01, L=1.0, device=DEV, seed=0)
dt = time.time() - t0
rows = np.asarray(rows)   # cols: z_set, z_obs, cascade_mean, cascade_std, gcc_mean, gcc_std

print(f"\n{'z_obs':>6} {'cascade':>8} {'gcc':>6}   window landmarks")
for r in rows:
    zo, cm, gm = r[1], r[2], r[4]
    mark = ""
    if abs(r[0]-4.51) < 0.01: mark = "<- z_c^RGG (percolation onset)"
    if abs(r[0]-7.0)  < 0.01: mark = "<- expected peak"
    if abs(r[0]-13.3) < 0.01: mark = "<- 2/phi (blocking boundary)"
    print(f"{zo:6.2f} {cm:8.3f} {gm:6.3f}   {mark}")

peak_i = int(np.argmax(rows[:,2]))
print(f"\npeak cascade={rows[peak_i,2]:.3f} at z_obs={rows[peak_i,1]:.2f}  (theory peak ~6-7)")
print(f"GPU: {dt:.1f}s for {len(zgrid)} z-points x B={B}  ({dt/len(zgrid)*1000:.0f} ms/z-point)", end="")
if DEV == "cuda":
    print(f"  | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB", end="")
print(f"\n=> full 40-point x B={B} run ~ {dt/len(zgrid)*40:.0f}s; window reproduces iff onset~4.5, peak~7, falloff by ~13")
