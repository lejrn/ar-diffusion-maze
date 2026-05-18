import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from maze import GridMaze


def test_bfs_finds_path():
    maze = GridMaze(size=8, seed=0)
    path = maze.bfs_path()
    assert path is not None
    assert path[0] == maze.start
    assert path[-1] == maze.goal


def test_path_avoids_walls():
    maze = GridMaze(size=8, seed=0)
    path = maze.bfs_path()
    for r, c in path:
        assert not maze.walls[r, c]


def test_random_path_returns_valid():
    import numpy as np

    maze = GridMaze(size=8, seed=0)
    rng = np.random.default_rng(1)
    path = maze.random_path(rng)
    assert path is not None
    assert len(path) >= 2
