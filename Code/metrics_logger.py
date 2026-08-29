"""
metrics_logger.py

MetricsLogger
--------------
Purpose: keep diagnostics separate from dynamics.

Tracks (per timestep):
  - Cascade size (activated fraction, and optional count)
  - Mean degree z(t)
  - Discovery rate (# seeds / encounters)
  - Resource depletion curves (masses per patch)

Designed to consume step outputs from SpatialCascadeSimulation.step().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetricsSnapshot:
    """
    Container for a single timestep's logged metrics.

    Fields are stored in simple numeric form for easy conversion to numpy arrays.
    """
    timestep: int
    cascade_fraction: float
    cascade_count: int
    mean_degree: float
    num_seeds: int
    masses: np.ndarray
    all_depleted: bool


class MetricsLogger:
    '''
    Collect and export time-series diagnostics for a SpatialCascadeSimulation run.

    Major steps:
      1) Record per-step metrics from simulation step() outputs.
      2) Provide numpy export for plotting/analysis.
      3) Provide lightweight summary statistics.

    Input parameters:
      - num_patches: Optional[int]
      - store_active_masks: bool (default False)
      - store_seed_indices: bool (default False)

    Return value:
      - None (logger stores metrics internally)
    '''

    def __init__(
        self,
        num_patches: Optional[int] = None,
        store_active_masks: bool = False,
        store_seed_indices: bool = False,
    ) -> None:
        self._num_patches_hint = num_patches
        self._store_active_masks = bool(store_active_masks)
        self._store_seed_indices = bool(store_seed_indices)

        self._snapshots: List[MetricsSnapshot] = []

        # Optional raw objects (can be memory-heavy)
        self._active_masks: List[np.ndarray] = []
        self._seed_indices: List[np.ndarray] = []

        # keep track of gcc size
        self._gcc_fraction = []

        # (store degrees per node and activation mask per timestep)
        self._degrees: List[np.ndarray] = []
        self._activated_for_degree: List[np.ndarray] = []


    # -----------------------------
    # Recording
    # -----------------------------

    def record_step(self, timestep: int, step_result: Any) -> None:
        '''
        Record a single timestep.

        Expected fields on step_result (duck-typed):
          - active_mask: np.ndarray[bool] shape (N,)
          - mean_degree: float
          - seed_indices: array-like[int]
          - masses_after: array-like[float] shape (K,)
          - all_depleted: bool

        Input parameters:
          - timestep: int
          - step_result: Any (typically the dataclass returned by SpatialCascadeSimulation.step())

        Return value:
          - None
        '''
        active_mask = np.asarray(step_result.active_mask, dtype=bool)
        seed_indices = np.asarray(step_result.seed_indices, dtype=int)
        masses = np.asarray(step_result.masses_after, dtype=float)

        # Degree per node for this timestep's induced graph (neighbors fixed within timestep)
        degrees = np.fromiter((len(nbrs) for nbrs in step_result.neighbors), dtype=int, count=len(step_result.neighbors))

        # Store for degree-conditioned activation analysis
        self._degrees.append(degrees)
        self._activated_for_degree.append(active_mask.copy())

        if self._num_patches_hint is not None and masses.size != int(self._num_patches_hint):
            raise ValueError(
                f"masses_after length {masses.size} != expected num_patches {self._num_patches_hint}"
            )

        cascade_count = int(active_mask.sum())
        cascade_fraction = float(active_mask.mean()) if active_mask.size > 0 else 0.0
        mean_degree = float(step_result.mean_degree)
        num_seeds = int(seed_indices.size)
        all_depleted = bool(step_result.all_depleted)

        gcc_fraction = self.compute_gcc_fraction_from_neighbors(step_result.neighbors)
        self._gcc_fraction.append(float(gcc_fraction))

        self._snapshots.append(
            MetricsSnapshot(
                timestep=int(timestep),
                cascade_fraction=cascade_fraction,
                cascade_count=cascade_count,
                mean_degree=mean_degree,
                num_seeds=num_seeds,
                masses=masses.copy(),
                all_depleted=all_depleted,
            )
        )

        if self._store_active_masks:
            self._active_masks.append(active_mask.copy())

        if self._store_seed_indices:
            self._seed_indices.append(seed_indices.copy())

    # -----------------------------
    # Accessors / export
    # -----------------------------

    def __len__(self) -> int:
        return len(self._snapshots)

    def is_empty(self) -> bool:
        return len(self._snapshots) == 0

    def to_numpy(self) -> Dict[str, np.ndarray]:
        '''
        Export logged metrics as numpy arrays.

        Return keys:
          - t: shape (T,)
          - cascade_fraction: shape (T,)
          - cascade_count: shape (T,)
          - mean_degree: shape (T,)
          - num_seeds: shape (T,)
          - masses: shape (T, K)
          - all_depleted: shape (T,)
          - gcc_fraction: shape (T,)
          - degrees: object array of length T, each element shape (N,)  (if available)
          - activated_for_degree: object array length T, each element shape (N,) bool (if available)

        Input parameters:
          - None

        Return value:
          - dict[str, np.ndarray]
        '''
        if self.is_empty():
            K = int(self._num_patches_hint) if self._num_patches_hint is not None else 0
            out = {
                "t": np.zeros((0,), dtype=int),
                "cascade_fraction": np.zeros((0,), dtype=float),
                "cascade_count": np.zeros((0,), dtype=int),
                "mean_degree": np.zeros((0,), dtype=float),
                "num_seeds": np.zeros((0,), dtype=int),
                "masses": np.zeros((0, K), dtype=float),
                "all_depleted": np.zeros((0,), dtype=bool),
                "gcc_fraction": np.zeros((0,), dtype=float),
            }
            # Degree-conditioned exports (empty)
            out["degrees"] = np.array([], dtype=object)
            out["activated_for_degree"] = np.array([], dtype=object)
            return out

        t = np.array([s.timestep for s in self._snapshots], dtype=int)
        cascade_fraction = np.array([s.cascade_fraction for s in self._snapshots], dtype=float)
        cascade_count = np.array([s.cascade_count for s in self._snapshots], dtype=int)
        mean_degree = np.array([s.mean_degree for s in self._snapshots], dtype=float)
        num_seeds = np.array([s.num_seeds for s in self._snapshots], dtype=int)
        all_depleted = np.array([s.all_depleted for s in self._snapshots], dtype=bool)

        masses = np.stack([s.masses for s in self._snapshots], axis=0)  # (T, K)
        gcc_fraction = np.asarray(self._gcc_fraction, dtype=float)

        out = {
            "t": t,
            "cascade_fraction": cascade_fraction,
            "cascade_count": cascade_count,
            "mean_degree": mean_degree,
            "num_seeds": num_seeds,
            "masses": masses,
            "all_depleted": all_depleted,
            "gcc_fraction": gcc_fraction,
        }

        # Degree-conditioned: store as object arrays (each timestep can be length N)
        if len(self._degrees) == len(self._snapshots):
            out["degrees"] = np.asarray(self._degrees, dtype=object)
        else:
            out["degrees"] = np.array([], dtype=object)

        if len(self._activated_for_degree) == len(self._snapshots):
            out["activated_for_degree"] = np.asarray(self._activated_for_degree, dtype=object)
        else:
            out["activated_for_degree"] = np.array([], dtype=object)

        return out

    def get_active_masks(self) -> List[np.ndarray]:
        '''
        Return stored active masks (if enabled).

        Input parameters:
          - None

        Return value:
          - list[np.ndarray]
        '''
        return list(self._active_masks)

    def get_seed_indices(self) -> List[np.ndarray]:
        '''
        Return stored seed indices (if enabled).

        Input parameters:
          - None

        Return value:
          - list[np.ndarray]
        '''
        return list(self._seed_indices)

        def get_degrees(self) -> List[np.ndarray]:
        '''
        Return stored per-node degrees for each timestep.

        Input parameters:
          - None

        Return value:
          - list[np.ndarray]
        '''
        return list(self._degrees)

    def get_activated_for_degree(self) -> List[np.ndarray]:
        '''
        Return stored activation masks used for degree-conditioned analysis.

        Input parameters:
          - None

        Return value:
          - list[np.ndarray]
        '''
        return list(self._activated_for_degree)


    # -----------------------------
    # Summaries
    # -----------------------------


    @staticmethod
    def compute_gcc_fraction_from_neighbors(neighbors) -> float:
        '''
        Compute giant connected component (GCC) fraction from an adjacency-list graph.

        Major steps:
          1) Run BFS/DFS to find connected components.
          2) Track the largest component size.
          3) Return largest_size / N.

        Input parameters:
          - neighbors: list[list[int]] adjacency lists (assumed undirected)

        Return value:
          - gcc_fraction: float in [0, 1]
        '''
        num_nodes = len(neighbors)
        if num_nodes == 0:
            return 0.0

        visited = np.zeros((num_nodes,), dtype=bool)
        largest = 0

        for start in range(num_nodes):
            if visited[start]:
                continue

            stack = [start]
            visited[start] = True
            component_size = 0

            while stack:
                v = stack.pop()
                component_size += 1
                for u in neighbors[v]:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)

            if component_size > largest:
                largest = component_size
            if largest == num_nodes:
                break

        return float(largest) / float(num_nodes)
    
    def summary(self) -> Dict[str, float]:
        '''
        Compute a small set of summary statistics over the logged time-series.

        Output fields:
          - steps
          - mean_cascade_fraction
          - max_cascade_fraction
          - mean_degree
          - mean_num_seeds
          - final_total_mass
          - depleted (1.0 or 0.0)

        Input parameters:
          - None

        Return value:
          - dict[str, float]
        '''
        if self.is_empty():
            return {
                "steps": 0.0,
                "mean_cascade_fraction": 0.0,
                "max_cascade_fraction": 0.0,
                "mean_degree": 0.0,
                "mean_num_seeds": 0.0,
                "final_total_mass": 0.0,
                "depleted": 0.0,
            }

        arrays = self.to_numpy()
        masses = arrays["masses"]
        final_total_mass = float(masses[-1].sum()) if masses.size > 0 else 0.0

        return {
            "steps": float(len(self)),
            "mean_cascade_fraction": float(arrays["cascade_fraction"].mean()),
            "max_cascade_fraction": float(arrays["cascade_fraction"].max()),
            "mean_degree": float(arrays["mean_degree"].mean()),
            "mean_num_seeds": float(arrays["num_seeds"].mean()),
            "final_total_mass": final_total_mass,
            "depleted": float(bool(arrays["all_depleted"][-1])),
        }

