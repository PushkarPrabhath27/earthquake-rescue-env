from __future__ import annotations

from collections import deque

import numpy as np


class TerrainSimulator:
    GRID_SIZE = 64
    START = (0, 2)

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, rubble_density: float = 0.25) -> np.ndarray:
        density = float(np.clip(rubble_density, 0.0, 0.75))
        grid = (self.rng.random((self.GRID_SIZE, self.GRID_SIZE)) < density).astype(np.int8)

        clear_rows = (0, 2, 16, 32, 48, 63)
        clear_cols = (0, 2, 16, 32, 48, 63)
        for row in clear_rows:
            grid[row, :] = 0
        for col in clear_cols:
            grid[:, col] = 0

        grid[0:3, 0:4] = 0
        grid[61:64, 61:64] = 0
        return grid

    @classmethod
    def reachable_cells(
        cls, grid: np.ndarray, start: tuple[int, int] | None = None
    ) -> list[tuple[int, int]]:
        origin = start or cls.START
        if grid[origin[0], origin[1]] != 0:
            return []

        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([origin])

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cls.GRID_SIZE and 0 <= ny < cls.GRID_SIZE and grid[nx, ny] == 0:
                    queue.append((nx, ny))

        return sorted(visited)
