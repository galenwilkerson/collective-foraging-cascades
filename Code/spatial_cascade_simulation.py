# spatial_cascade_simulation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from particle_system_2d import ParticleSystem2D
from resource_patch import ResourceField
from spatial_cascade_ltm import SpatialCascadeLTM


@dataclass
class SpatialCascadeStepResult:
    '''
    Container for per-timestep outputs (useful for logging/diagnostics).

    Fields:
      ...
      - mean_degree: float observed mean degree of the induced RGG for this timestep
      - neighbors: list[list[int]] adjacency lists for the induced graph G_t (built at start of timestep)
    '''
    t: int
    seed_indices: np.ndarray
    active_mask: np.ndarray
    parents: np.ndarray
    activation_order: np.ndarray
    newly_active_mask: np.ndarray
    masses_after: np.ndarray
    all_depleted: bool
    mean_degree: float
    neighbors: list[list[int]]


class SpatialCascadeSimulation:
    '''
    Orchestration layer implementing the paper's per-timestep update rule.

    Major steps per timestep:
      1) Build induced interaction graph G_t from current particle positions (fixed within the timestep)
      2) Random-walk particles (event-local activation is reset each timestep, so all particles are "inactive")
      3) Detect resource encounters to form seed set
      4) Run LTM cascade on G_t to fixpoint
      5) Move newly-activated particles toward their activation parents
      6) Consume resource mass based on active particles
      7) Reset activity flags implicitly by discarding active_mask (event-local activation)
      8) Stop when all resources are depleted (or after T steps)

    Input parameters:
      - particles: ParticleSystem2D
      - resources: ResourceField
      - cascade: SpatialCascadeLTM
      - threshold: float (paper's activation threshold parameter)
      - step_size: float (Δ)
      - max_steps: int (T)
      - rng: np.random.Generator or None (if None, a default generator is created)

    Return value:
      - None (state lives in held components; step() returns a SpatialCascadeStepResult)
    '''

    def __init__(
        self,
        particles: ParticleSystem2D,
        resources: ResourceField,
        cascade: SpatialCascadeLTM,
        threshold: float,
        step_size: float,
        max_steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.particles = particles
        self.resources = resources
        self.cascade = cascade

        self.threshold = float(threshold)
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)

        self.rng = rng if rng is not None else np.random.default_rng()

        self.t = 0  # current timestep index

    @staticmethod
    def _bool_list_to_indices(bool_list) -> np.ndarray:
        '''
        Convert a list[bool] (or np.ndarray[bool]) into integer indices.

        Input parameters:
          - bool_list: array-like of bool

        Return value:
          - indices: np.ndarray[int]
        '''
        mask = np.asarray(bool_list, dtype=bool)
        return np.nonzero(mask)[0].astype(int)

    @staticmethod
    def _compute_mean_degree(neighbors) -> float:
        '''
        Compute mean degree from a neighbor-list representation.

        Input parameters:
          - neighbors: list[list[int]]

        Return value:
          - mean_degree: float
        '''
        if len(neighbors) == 0:
            return 0.0
        degs = np.fromiter((len(nbrs) for nbrs in neighbors), dtype=float, count=len(neighbors))
        return float(degs.mean())

    def step(self) -> SpatialCascadeStepResult:
        '''
        Execute one full timestep of the paper's update rule.

        Major steps:
          1) Build neighbors on current positions (graph fixed within timestep)
          2) Random-walk all particles (since activation is event-local each step)
          3) Detect seeds (resource encounters)
          4) Run LTM cascade to fixpoint
          5) Directed move for newly-active (non-seed active nodes) toward parents
          6) Consume resources based on final active set
          7) Return diagnostics

        Input parameters:
          - None

        Return value:
          - SpatialCascadeStepResult
        '''
        t_now = int(self.t)

        # 1) Build induced interaction graph on current positions (G_t fixed for this timestep)
        neighbors = self.cascade.build_neighbor_lists(self.particles.positions)
        mean_degree = self._compute_mean_degree(neighbors)

        # 2) Random-walk "inactive" particles.
        #    Event-local activation resets each step => everyone starts inactive.
        inactive_mask = np.ones((self.particles.positions.shape[0],), dtype=bool)
        self.particles.random_walk_step(inactive_mask=inactive_mask, step_size=self.step_size, rng=self.rng)

        # 3) Detect resource encounters => seed set
        seed_bool = self.resources.detect_resources(self.particles.positions)
        seed_indices = self._bool_list_to_indices(seed_bool)

        # 4) Run cascade to fixpoint on fixed neighbors
        active_mask, parents, activation_order = self.cascade.run_cascade(
            neighbors=neighbors,
            seed_indices=seed_indices.tolist(),
            threshold=self.threshold,
            rng=self.rng,
        )

        # 5) Move newly activated particles toward parents (exclude seeds)
        seed_mask = np.zeros_like(active_mask, dtype=bool)
        if seed_indices.size > 0:
            seed_mask[seed_indices] = True
        newly_active_mask = np.asarray(active_mask, dtype=bool) & (~seed_mask)

        self.particles.directed_step_toward(
            parents=np.asarray(parents, dtype=int),
            newly_active_mask=newly_active_mask,
            step_size=self.step_size,
        )

        # Safety wrap (in case motion methods don't wrap internally)
        self.particles.wrap_positions()

        # 6) Consume resources based on the final active set
        self.resources.consume(positions=self.particles.positions, active_mask=np.asarray(active_mask, dtype=bool))

        masses_after = np.array([p.mass for p in self.resources.patches], dtype=float)
        depleted = bool(self.resources.all_depleted())

        # 7) event-local activation reset happens by not storing active state across steps
        self.t += 1

        return SpatialCascadeStepResult(
            t=t_now,
            seed_indices=seed_indices,
            active_mask=np.asarray(active_mask, dtype=bool),
            parents=np.asarray(parents, dtype=int),
            activation_order=np.asarray(activation_order, dtype=int) if activation_order is not None else np.array([], dtype=int),
            newly_active_mask=newly_active_mask,
            masses_after=masses_after,
            all_depleted=depleted,
            mean_degree=mean_degree,
            neighbors=neighbors,
        )

    
    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        '''
        Run the simulation loop until resources are depleted or step limit reached.

        Major steps:
          1) Iterate step()
          2) Stop early if all resources depleted
          3) Return a compact history dict for analysis

        Input parameters:
          - max_steps: Optional[int] overrides self.max_steps if provided

        Return value:
          - history: dict with time series arrays
        '''
        T = int(self.max_steps if max_steps is None else max_steps)

        cascade_sizes = []
        mean_degrees = []
        num_seeds = []
        masses = []

        for _ in range(T):
            result = self.step()

            cascade_sizes.append(float(result.active_mask.mean()))
            mean_degrees.append(float(result.mean_degree))
            num_seeds.append(int(result.seed_indices.size))
            masses.append(result.masses_after.copy())

            if result.all_depleted:
                break

        masses_arr = np.asarray(masses, dtype=float) if len(masses) > 0 else np.zeros((0, len(self.resources.patches)), dtype=float)

        return {
            "cascade_size_mean_frac": np.asarray(cascade_sizes, dtype=float),
            "mean_degree": np.asarray(mean_degrees, dtype=float),
            "num_seeds": np.asarray(num_seeds, dtype=int),
            "masses": masses_arr,
            "stopped_because_depleted": bool(self.resources.all_depleted()),
            "steps_run": int(len(cascade_sizes)),
        }

