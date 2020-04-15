import os

import numpy as np

from utils import PROJECT_BASE_DIR

from env.mazeworld.maze_world_generator import MazeWorldGenerator

CACHE_DIR: str = os.path.join(
    PROJECT_BASE_DIR,
    'env', 'mazeworld', 'mazeworld-caches')

class MazeWorldCache:
    def __init__(self, generator: MazeWorldGenerator):
        self.generator = generator 

    def _get_cache_name(self, rand_seed: int) -> str:
        return f"{self.generator.name()}_{rand_seed}.csv.npy"

    def _get_cache_path(self, rand_seed: int) -> str:
        return os.path.join(
            CACHE_DIR, 
            self._get_cache_name(rand_seed))

    def _cache_exists(self, rand_seed: int) -> bool:
        return os.path.exists(self._get_cache_path(rand_seed))

    def _get_cached_board(self, rand_seed: int) -> np.ndarray:
        if self._cache_exists(rand_seed):
            return np.load(self._get_cache_path(rand_seed))
        else:
            fpath = self._get_cache_path(rand_seed)
            fpath = fpath[:-4] #remove .npy at end, 
            print(fpath)
            # because numpy automatically adds it
            board: np.ndarray = self.generator.make(rand_seed)
            np.save(fpath, board)
            return board

