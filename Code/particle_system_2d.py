import numpy as np


class ParticleSystem2D:
    '''
    ParticleSystem2D: motion-only particle state on a 2D torus.

    Major responsibilities:
      1) Maintain particle positions in [0, domain_size) x [0, domain_size).
      2) Apply unbiased random-walk steps to inactive particles.
      3) Apply directed steps toward activation parents for newly active particles.
      4) Enforce periodic boundary conditions via torus wrapping.

    Notes:
      - This class intentionally does NOT know about cascades, resources, or thresholds.
      - Directed motion uses minimum-image displacement on the torus.
      - Default directed motion is a continuous step of length `step_size` toward the parent.
        (If you later want discrete 4-neighbor grid steps, add a policy flag.)

    Input parameters (constructor):
      - initial_positions: np.ndarray, shape (N, 2)
      - domain_size: float

    Public attributes:
      - positions: np.ndarray, shape (N, 2)
      - domain_size: float

    Return value:
      - None
    '''

    def __init__(self, initial_positions: np.ndarray, domain_size: float = 1.0):
        self.positions = np.asarray(initial_positions, dtype=float).copy()
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError(f"initial_positions must have shape (N, 2). Got {self.positions.shape}")
        if domain_size <= 0.0:
            raise ValueError("domain_size must be positive.")
        self.domain_size = float(domain_size)
        self.wrap_positions()

    def wrap_positions(self) -> None:
        '''
        Wrap all particle positions into the fundamental domain [0, domain_size).

        Major steps:
          1) Apply modulo operation component-wise.

        Input parameters:
          - None

        Return value:
          - None
        '''
        self.positions %= self.domain_size

    def random_walk_step(self, inactive_mask: np.ndarray, step_size: float, rng: np.random.Generator) -> None:
        '''
        Apply an unbiased random-walk step to particles marked inactive.

        Major steps:
          1) Sample random angles for inactive particles.
          2) Convert to unit direction vectors.
          3) Add step_size * direction to those particles.
          4) Wrap positions on the torus.

        Input parameters:
          - inactive_mask: np.ndarray[bool], shape (N,)
          - step_size: float
          - rng: np.random.Generator

        Return value:
          - None
        '''
        inactive_mask = np.asarray(inactive_mask, dtype=bool)
        if inactive_mask.shape != (self.positions.shape[0],):
            raise ValueError(f"inactive_mask must have shape (N,). Got {inactive_mask.shape}")

        if step_size < 0.0:
            raise ValueError("step_size must be nonnegative.")

        inactive_indices = np.where(inactive_mask)[0]
        if inactive_indices.size == 0:
            return

        angles = rng.uniform(0.0, 2.0 * np.pi, size=inactive_indices.size)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        self.positions[inactive_indices] += float(step_size) * directions
        self.wrap_positions()

    def directed_step_toward(
        self,
        parents: np.ndarray,
        newly_active_mask: np.ndarray,
        step_size: float,
    ) -> None:
        '''
        Move newly activated particles one step toward their activating parent.

        Uses minimum-image displacement on the torus:
          displacement = parent - child, mapped into [-L/2, L/2) per coordinate.

        Major steps:
          1) Select indices where newly_active_mask is True.
          2) For each, read parent index; skip if parent < 0.
          3) Compute minimum-image displacement vector toward parent.
          4) Normalize displacement to unit direction (if nonzero).
          5) Move child by step_size in that direction.
          6) Wrap positions.

        Input parameters:
          - parents: np.ndarray[int], shape (N,), parents[i] is activating neighbor index (or -1)
          - newly_active_mask: np.ndarray[bool], shape (N,)
          - step_size: float

        Return value:
          - None
        '''
        parents = np.asarray(parents)
        newly_active_mask = np.asarray(newly_active_mask, dtype=bool)

        if parents.shape != (self.positions.shape[0],):
            raise ValueError(f"parents must have shape (N,). Got {parents.shape}")
        if newly_active_mask.shape != (self.positions.shape[0],):
            raise ValueError(f"newly_active_mask must have shape (N,). Got {newly_active_mask.shape}")
        if step_size < 0.0:
            raise ValueError("step_size must be nonnegative.")

        active_indices = np.where(newly_active_mask)[0]
        if active_indices.size == 0:
            return

        # Minimum-image displacement: map into [-L/2, L/2)
        L = self.domain_size

        for i in active_indices:
            parent_idx = int(parents[i])
            if parent_idx < 0:
                continue

            disp = self.positions[parent_idx] - self.positions[i]
            disp = np.where(disp > 0.5 * L, disp - L, disp)
            disp = np.where(disp < -0.5 * L, disp + L, disp)

            norm = float(np.linalg.norm(disp))
            if norm == 0.0:
                continue

            direction = disp / norm
            self.positions[i] += float(step_size) * direction

        self.wrap_positions()

