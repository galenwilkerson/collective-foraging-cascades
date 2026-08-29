import numpy as np
from typing import List, Sequence, Tuple, Optional


class SpatialCascadeLTM:
    '''
    SpatialCascadeLTM: Pure cascade engine for spatial LTM cascades on an induced random geometric graph (RGG).

    Major steps:
      1) Build neighbor lists from particle positions using an interaction radius r on a 2D torus.
      2) Run synchronous linear-threshold updates (fraction of active neighbors >= threshold) to a fixed point.
      3) Record one activating "parent" neighbor for each newly activated node at the moment it flips.

    Input parameters (constructor):
      - interaction_radius: float
      - domain_size: float (assumes square torus [0, domain_size)^2)
      - deterministic_parent: bool (choose first activating neighbor if True; otherwise use RNG)

    Public methods:
      - build_neighbor_lists(positions) -> neighbors
      - run_cascade(neighbors, seed_indices, threshold, rng=None) -> (active_mask, activation_parents, newly_activated_order)

    Returns:
      - active_mask: np.ndarray[bool] of shape (N,)
      - activation_parents: np.ndarray[int] of shape (N,), parent index or -1 if seed/never activated
      - newly_activated_order: List[int], activation order excluding seeds
    '''

    def __init__(
        self,
        interaction_radius: float,
        domain_size: float = 1.0,
        deterministic_parent: bool = True,
    ) -> None:
        self.interaction_radius = float(interaction_radius)
        self.domain_size = float(domain_size)
        self.deterministic_parent = bool(deterministic_parent)

    def _torus_displacement_2d(self, point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        """
        Compute minimal displacement vector from point_a to point_b on a 2D torus.

        Parameters:
          - point_a: np.ndarray shape (2,)
          - point_b: np.ndarray shape (2,)

        Returns:
          - displacement: np.ndarray shape (2,)
        """
        displacement = point_b - point_a
        half_side = 0.5 * self.domain_size
        displacement = np.where(displacement > half_side, displacement - self.domain_size, displacement)
        displacement = np.where(displacement < -half_side, displacement + self.domain_size, displacement)
        return displacement

    def build_neighbor_lists(self, positions: np.ndarray) -> List[List[int]]:
        """
        Build undirected neighbor lists from particle positions using an interaction radius on a 2D torus.

        Parameters:
          - positions: np.ndarray of shape (N, 2), positions in [0, domain_size)

        Returns:
          - neighbors: List[List[int]] length N, neighbors[i] lists indices j != i with torus distance < r
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must be an array of shape (N, 2).")

        num_particles = positions.shape[0]
        neighbors: List[List[int]] = [[] for _ in range(num_particles)]
        radius_squared = self.interaction_radius ** 2

        # Brute-force O(N^2) neighbor building (skeleton; optimize later if needed)
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                displacement = self._torus_displacement_2d(positions[i], positions[j])
                distance_squared = float(displacement[0] * displacement[0] + displacement[1] * displacement[1])
                if distance_squared < radius_squared:
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        return neighbors


    @staticmethod
    def compute_gcc_fraction_from_neighbors(neighbors) -> float:
        '''
        Compute giant connected component (GCC) fraction from an adjacency-list graph.

        Major steps:
          1) Run DFS to find connected components.
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

    def _select_parent(
        self,
        activating_neighbors: Sequence[int],
        rng: Optional[np.random.Generator],
    ) -> int:
        """
        Select one activating neighbor as parent.

        Parameters:
          - activating_neighbors: indices of currently active neighbors
          - rng: optional np.random.Generator (used if deterministic_parent is False)

        Returns:
          - parent_index: int
        """
        if len(activating_neighbors) == 0:
            raise ValueError("Cannot select a parent from an empty list of activating neighbors.")

        if self.deterministic_parent:
            return int(activating_neighbors[0])

        if rng is None:
            rng = np.random.default_rng()
        return int(activating_neighbors[int(rng.integers(0, len(activating_neighbors)))])

    def run_cascade(
        self,
        neighbors: List[List[int]],
        seed_indices: Sequence[int],
        threshold: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Run a synchronous linear-threshold cascade to a fixed point on a provided neighbor graph.

        Update rule:
          - inactive node i activates if fraction of active neighbors >= threshold
          - if degree(i) == 0, define fraction(i) = 0

        Parent recording:
          - when i flips inactive -> active, record activation_parents[i] = chosen active neighbor
          - seeds have parent = -1

        Parameters:
          - neighbors: List[List[int]] length N
          - seed_indices: iterable of seed node indices to activate initially
          - threshold: float in [0, 1]
          - rng: optional np.random.Generator for non-deterministic parent selection

        Returns:
          - active_mask: np.ndarray[bool] shape (N,)
          - activation_parents: np.ndarray[int] shape (N,), -1 if seed or never activated
          - newly_activated_order: List[int] activation order excluding seeds
        """
        num_nodes = len(neighbors)
        if num_nodes == 0:
            return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=int), []

        threshold_value = float(threshold)
        if not (0.0 <= threshold_value <= 1.0):
            raise ValueError("threshold must be in [0, 1].")

        active_mask = np.zeros((num_nodes,), dtype=bool)
        activation_parents = np.full((num_nodes,), -1, dtype=int)
        newly_activated_order: List[int] = []

        # Initialize seeds
        for seed in seed_indices:
            seed_int = int(seed)
            if seed_int < 0 or seed_int >= num_nodes:
                raise IndexError(f"Seed index {seed_int} out of bounds for N={num_nodes}.")
            active_mask[seed_int] = True

        # Synchronous rounds until convergence
        changed = True
        while changed:
            changed = False
            new_nodes: List[int] = []
            new_parents: List[int] = []

            for i in range(num_nodes):
                if active_mask[i]:
                    continue

                neighbor_list = neighbors[i]
                degree = len(neighbor_list)
                if degree == 0:
                    continue  # fraction = 0

                active_neighbors = [j for j in neighbor_list if active_mask[j]]
                active_fraction = len(active_neighbors) / degree

                if active_fraction >= threshold_value and len(active_neighbors) > 0:
                    new_nodes.append(i)
                    new_parents.append(self._select_parent(active_neighbors, rng))

            if len(new_nodes) > 0:
                changed = True
                for node, parent in zip(new_nodes, new_parents):
                    active_mask[node] = True
                    activation_parents[node] = parent
                    newly_activated_order.append(node)

        return active_mask, activation_parents, newly_activated_order


def run_spatial_cascade_ltm_smoke_tests() -> None:
    '''
    Smoke tests for SpatialCascadeLTM.

    Major steps:
      1) Test cascade behavior on a small synthetic chain graph.
      2) Test torus neighbor building across periodic boundaries.
    '''
    # Test 1: chain 0-1-2-3, seed at 0, threshold 0.5 -> full activation
    neighbors_chain = [[1], [0, 2], [1, 3], [2]]
    engine = SpatialCascadeLTM(interaction_radius=0.2, deterministic_parent=True)  # r unused for this test

    active_mask, parents, order = engine.run_cascade(
        neighbors=neighbors_chain,
        seed_indices=[0],
        threshold=0.5,
    )

    assert active_mask.tolist() == [True, True, True, True], "Chain should fully activate at threshold 0.5."
    assert parents[0] == -1, "Seed parent should be -1."
    assert parents[1] == 0, "Node 1 should be activated by node 0 in this chain."
    assert parents[2] == 1, "Node 2 should be activated by node 1 in this chain."
    assert parents[3] == 2, "Node 3 should be activated by node 2 in this chain."
    assert order == [1, 2, 3], "Activation order should follow the chain deterministically."

    # Test 2: torus boundary wrap check
    engine2 = SpatialCascadeLTM(interaction_radius=0.05, domain_size=1.0, deterministic_parent=True)
    positions = np.array([
        [0.99, 0.50],
        [0.01, 0.50],
        [0.50, 0.50],
    ])

    nbrs = engine2.build_neighbor_lists(positions)

    assert 1 in nbrs[0] and 0 in nbrs[1], "Points across boundary should be neighbors on torus."
    assert 2 not in nbrs[0] and 2 not in nbrs[1], "Center point should not be neighbor for r=0.05."

    print("SpatialCascadeLTM smoke tests: PASSED")


def main():
    run_spatial_cascade_ltm_smoke_tests()


if __name__ == "__main__":
    main()
