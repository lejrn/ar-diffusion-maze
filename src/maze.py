from collections import deque

import numpy as np


class GridMaze:
    def __init__(self, size=10, seed=42):
        rng = np.random.default_rng(seed)
        self.size = size
        self.walls = rng.random((size, size)) < 0.2
        self.walls[0, :] = False
        self.walls[:, size - 1] = False
        self.walls[0, 0] = False
        self.walls[size - 1, size - 1] = False
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

    def bfs_path(self, start=None, goal=None):
        start = start or self.start
        goal = goal or self.goal
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.size
                    and 0 <= nc < self.size
                    and not self.walls[nr, nc]
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

        return None

    def random_path(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(200):
            r1, c1 = rng.integers(0, self.size, size=2)
            r2, c2 = rng.integers(0, self.size, size=2)
            if (
                not self.walls[r1, c1]
                and not self.walls[r2, c2]
                and (r1, c1) != (r2, c2)
            ):
                path = self.bfs_path((r1, c1), (r2, c2))
                if path and len(path) >= 3:
                    return path

        return self.bfs_path()
