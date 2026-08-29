import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ResourcePatch:
    '''
    A single localized resource region with finite mass.

    Parameters:
      - center_xy: np.ndarray shape (2,), center location in [0, domain_size)
      - radius: float, footprint radius (disc)
      - mass: float, remaining resource mass (>= 0)
    '''
    center_xy: np.ndarray
    radius: float
    mass: float


class ResourceField:
    '''
    ResourceField: resources only (detection + depletion bookkeeping).

    Major steps (per usage):
      1) detect_resources(): identify which particles intersect any non-depleted patch
      2) consume(): decrement patch mass based on active particles occupying patches
      3) all_depleted(): report completion

    Notes:
      - Domain is a 2D torus (periodic boundary conditions).
      - Resources are discs with (center, radius, mass).
      - No cascade logic and no motion logic live here.
    '''

    def __init__(
        self,
        patches: List[ResourcePatch],
        domain_size: float = 1.0,
        consumption_rate_per_particle: float = 1.0,
        depletion_epsilon: float = 0.0,
    ):
        '''
        Initialize a ResourceField.

        Input parameters:
          - patches: list[ResourcePatch]
          - domain_size: float
          - consumption_rate_per_particle: float, mass consumed per active particle per call to consume()
          - depletion_epsilon: float, treat mass <= depletion_epsilon as depleted

        Return value:
          - None
        '''
        if domain_size <= 0:
            raise ValueError("domain_size must be positive.")
        if consumption_rate_per_particle < 0:
            raise ValueError("consumption_rate_per_particle must be nonnegative.")
        if depletion_epsilon < 0:
            raise ValueError("depletion_epsilon must be nonnegative.")

        self._domain_size = float(domain_size)
        self._consumption_rate = float(consumption_rate_per_particle)
        self._depletion_epsilon = float(depletion_epsilon)

        # Normalize centers into [0, domain_size)
        normalized_patches: List[ResourcePatch] = []
        for patch in patches:
            center_xy = np.asarray(patch.center_xy, dtype=float).reshape(2,)
            center_xy = np.mod(center_xy, self._domain_size)
            normalized_patches.append(
                ResourcePatch(center_xy=center_xy, radius=float(patch.radius), mass=float(patch.mass))
            )
        self._patches = normalized_patches

    @property
    def patches(self) -> List[ResourcePatch]:
        '''
        Return the list of patches (mutable objects). Use carefully.

        Return value:
          - list[ResourcePatch]
        '''
        return self._patches

    @property
    def domain_size(self) -> float:
        '''
        Return the domain size.

        Return value:
          - float
        '''
        return self._domain_size

    def _minimum_image_displacement(self, source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        '''
        Compute minimum-image displacement on a 2D torus.

        Input parameters:
          - source_xy: np.ndarray shape (2,)
          - target_xy: np.ndarray shape (2,)

        Return value:
          - disp: np.ndarray shape (2,)
        '''
        disp = target_xy - source_xy
        half = 0.5 * self._domain_size
        disp = np.where(disp > half, disp - self._domain_size, disp)
        disp = np.where(disp < -half, disp + self._domain_size, disp)
        return disp

    def detect_resources(self, positions: np.ndarray) -> List[bool]:
        '''
        Detect whether each particle intersects any non-depleted resource patch.

        Major steps:
          1) For each particle position, check each patch that is not depleted.
          2) Return a boolean list per particle: True if inside any patch footprint.

        Input parameters:
          - positions: np.ndarray shape (N, 2)

        Return value:
          - detected: list[bool] length N
        '''
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2).")

        num_particles = positions.shape[0]
        detected = [False] * num_particles

        for i in range(num_particles):
            p_xy = positions[i]
            for patch in self._patches:
                if patch.mass <= self._depletion_epsilon:
                    continue
                disp = self._minimum_image_displacement(patch.center_xy, p_xy)
                if float(disp[0] * disp[0] + disp[1] * disp[1]) <= patch.radius * patch.radius:
                    detected[i] = True
                    break

        return detected

    def consume(self, positions: np.ndarray, active_mask: np.ndarray) -> None:
        '''
        Reduce resource mass based on active particles located on resources.

        Major steps:
          1) Identify active particles.
          2) For each active particle, find all non-depleted patches it intersects.
          3) Decrement patch mass by consumption_rate_per_particle (clamped at 0).

        Input parameters:
          - positions: np.ndarray shape (N, 2)
          - active_mask: np.ndarray[bool] shape (N,)

        Return value:
          - None
        '''
        positions = np.asarray(positions, dtype=float)
        active_mask = np.asarray(active_mask, dtype=bool)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2).")
        if active_mask.ndim != 1 or active_mask.shape[0] != positions.shape[0]:
            raise ValueError("active_mask must have shape (N,) matching positions.")

        active_indices = np.where(active_mask)[0]
        if active_indices.size == 0:
            return

        if self._consumption_rate == 0.0:
            return

        for idx in active_indices:
            p_xy = positions[idx]
            for patch in self._patches:
                if patch.mass <= self._depletion_epsilon:
                    continue
                disp = self._minimum_image_displacement(patch.center_xy, p_xy)
                if float(disp[0] * disp[0] + disp[1] * disp[1]) <= patch.radius * patch.radius:
                    patch.mass = max(0.0, patch.mass - self._consumption_rate)

    def all_depleted(self) -> bool:
        '''
        Check whether all resources are depleted.

        Return value:
          - bool: True iff every patch has mass <= depletion_epsilon
        '''
        for patch in self._patches:
            if patch.mass > self._depletion_epsilon:
                return False
        return True

    @staticmethod
    def sample_uniform_patches(
        num_patches: int,
        radius: float,
        mass: float,
        domain_size: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "ResourceField":
        '''
        Convenience constructor: sample patches uniformly at random.

        Major steps:
          1) Sample centers uniformly in [0, domain_size)^2.
          2) Create ResourcePatch objects with shared radius and mass.

        Input parameters:
          - num_patches: int
          - radius: float
          - mass: float
          - domain_size: float
          - rng: Optional[np.random.Generator]

        Return value:
          - ResourceField
        '''
        if rng is None:
            rng = np.random.default_rng()

        centers = rng.random((num_patches, 2)) * float(domain_size)
        patches = [
            ResourcePatch(center_xy=centers[i], radius=float(radius), mass=float(mass))
            for i in range(num_patches)
        ]
        return ResourceField(
            patches=patches,
            domain_size=float(domain_size),
            consumption_rate_per_particle=1.0,
            depletion_epsilon=0.0,
        )

