# Part I figure pipeline (Chaos paper)

These are the scripts that generate the figures and sweep data for Part I
("A self-organizing mechanism for collective foraging: recruitment cascades couple
Lévy search to a clustered maximum-entropy resource field"), run on the corrected
simulation engine (`../Code/mobile_cascade.py`).

## Layout and paths

- `../Code/` — the simulation engine (`mobile_cascade.py`) and helper modules.
- `experiments/` — these figure/data scripts.
- `../figures/` — where the scripts write output PDFs.
- data — **not shipped** (see below).

Paths are resolved by `_paths.py`, which puts `../Code` on `sys.path` and defines
`FIG` (`../figures`) and `DATA`. There are no hardcoded machine paths; just run the
scripts from this directory.

## Data is not included

Precomputed sweep/replot caches (`.npz`) are not stored in the repository. Scripts
that read a cache look under `DATA`, which defaults to `<repo>/data`. Either:

- regenerate the data with the compute scripts (these run the engine), or
- set `FLOCKING_DATA` to a directory holding the `.npz` files:
  `FLOCKING_DATA=/path/to/data python replot_spectrum_v15.py`.

## Which script makes which figure

| Figure | Compute (writes data) | Plot (writes PDF) |
|---|---|---|
| Fig 1b cycle phase portrait | `../Code/gen_cycle_data.py` → `cycle_states.npz` | `PATCHES=1 python plot_cycle_statecolor_v15.py` |
| Fig 2 inverted-U window | — | `../Code/gen_invertedu_selftune_fig.py` |
| Fig 3 untuned finite-size collapse | `fss_cell.py` / `fss_aggregate.py` | `plot_collapse_v15.py` |
| Fig 4 coupling (coupled vs uncoupled) | `gen_q3_ablation_v15.py` | `replot_q3_v15.py` |
| Fig 5 harvest phase diagram | `gen_q4_phasegrid_v15.py` | `plot_phasegrid_v15.py` |
| Fig 6 optimal exponent vs D_f | — | `replot_caliber_v15.py` |
| Fig 7 activity spectrum | `rq2_spectrum.py` → `spectrum.npz` | `replot_spectrum_v15.py` |

SI figures: `gen_scalefree_v15.py`, `gen_levy_field_v15.py`,
`gen_susceptibility_cuda.py` + `plot_susceptibility_v15.py`,
`plot_recruit_cluster_v15.py`, `plot_frontspeed_v15.py`, `plot_supp_expts_v15.py`.

GPU (CUDA) is used by the compute scripts (`gen_*`, `rq2_spectrum`, `fss_cell`);
the `plot_*`/`replot_*` scripts are CPU-only and read cached data.
