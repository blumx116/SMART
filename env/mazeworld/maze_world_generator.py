import os
from typing import List, Iterable

import matplotlib.pyplot as plt 
import numpy as np

from utils import (
    flatmap, array_unique, array_contains, 
    array_random_choice, array_shuffle, PROJECT_BASE_DIR)

MAZEWORLD_CACHE_DIR: str = os.path.join(PROJECT_BASE_DIR, "env", "mazeworld", "gridworld-caches"))

class MazeWorldGenerator:
    tile_types = ["Wall", "Lava", "Empty"]

    def __init__(self, y_dim: int, x_dim: int, random_wall: int, n_wall: int, n_lava: int, random:):
        """
            Class for creating new gridworld environments of given format. 
            y_dim : number of grid tiles in the y direction
            x_dim : number of grid tiles in the x direction
            random_wall : number of wall tiles to place randomly before growing walls
            n_wall : total number of wall tiles to try to place
            n_lava : total number of lava tiles to try to place
        """
        self.x_dim : int = x_dim 
        self.y_dim : int = y_dim 
        self.random_wall = random_wall
        self.n_wall :int = n_wall
        self.n_lava :int = n_lava
        self.random = None 
        self._clear_state()

    def new(self) -> np.ndarray:
        self._randomize_state()
        return self.state()

    def state(self):
        return self._game_state


    def _clear_state(self) -> None:
        """
            sets the board state to a clear board of all 0's
        """
        self._game_state = np.zeros((self.y_dim+2, self.x_dim+2, 2))
        # add 2 to each dimension because we add a row of walls
        self._add_boundary_walls()

    def _add_boundary_walls(self, grid: np.ndarray) -> np.ndarray:
        """
            Adds walls to all tiles on the boundary. Walls are added in new grid tiles
            along boundary
            grid: np.ndarray[bool] : [y_dim, x_dim, 2], grid to be added to
            returns : np.ndarray[bool] : [y_dim+2, x_dim+2, 2], grid with walls added
        """
        shape: Tuple[int] = (grid.shape[0] + 2, grid.shape[1] + 2, ) + grid.shape[2:]
        new_grid = np.full(shape, False, bool) #np.ndarray[bool] : [y_dim+2, x_dim+2, 2]
        new_grid[1:-1, 1:-1, :] = grid #copy old values
        new_grid[0,:,0] = True #Top Wall
        new_grid[:,0,0] = True #Left Wall
        new_grid[-1,:,0] =True #Bottom Wall
        new_grid[:,-1,0] =True #Right Wall
        return new_grid

    def _extend_tree(self, tree: List[np.ndarray], required_type: str=None) -> np.ndarray:
        """
            will find a point that is adjacent to one of the points on the tree
            but does not break the tree property when added to the list. Choice 
            made at random

            tree: List of points, List[np.ndarray[int]] : [2,] (y, x)
            required_type : str, in Gridworld.tile_types or None
                will return a point of that type. If None, will 
                place no stipulations
            returns : np.ndarray[int] : [2,]  (y,x) new point to add
        """
        if len(tree) == 0:
            return self._random_point(required_type)
            # no restrictions in this case
        neighbors: List[np.ndarray] = flatmap(map(self._get_neighboring_tiles, tree))
        if required_type is not None:
            neighbors: Iterable[np.ndarray] = filter(lambda el : self._get_tile_type(el) == required_type, neighbors)

        #don't want to add someone already in tree

        neighbors = array_shuffle(neighbors)

        while len(neighbors) > 0:
            first: np.ndarray = neighbors.pop()
            if (self._get_tile_type(first) == required_type):
                if (not array_contains(first, tree)) and (not self.creates_cycle(first, tree)):
                    return first
        return None

    def creates_cycle(self, el: np.ndarray, tree: List[np.ndarray], block_type: str = "Wall") -> bool:
        # can create a cycle
        all_neighbors: List[np.ndarray] = list(filter( 
            lambda el : self._tile_is_of_type(el, block_type),
            self._get_neighboring_tiles(el, include_diagonals=True)))
        if len(all_neighbors) > 3: # this line not needed, just speed up
            return True
        strict_neighbors : List[np.ndarray] = list(filter(
            lambda el: self._tile_is_of_type(el, block_type),
            self._get_neighboring_tiles(el, include_diagonals=False)))
        if len(strict_neighbors) > 1: # just a speed up
            return True
        # now, we can technically return true if all 3 blocks are strictly contiguous
        strict_neighbor: np.ndarray = strict_neighbors[0]
        # strict_neighbor's strict neighbors form strictly contiguous block
        contiguous_block = [strict_neighbor] + self._get_neighboring_tiles(strict_neighbor)
        for neighbor in all_neighbors:
            if not array_contains(neighbor, contiguous_block):
                return True 
        return False

    def _add_lava(self, n_tiles: int) -> None:
        for _ in range(n_tiles):
            point: np.ndarray = self._random_point("Empty")
            self._game_state[point[0], point[1], 1] = 1.
            

    def _tile_is_of_type(self, point: np.ndarray, required_type: str=None) -> bool:
        if required_type is None:
            return True
        else:
            return self._get_tile_type(point) == required_type
        

    def _randomize_state(self) -> None:
        """
            randomizes the state to create a maze, such that there are no cycles
        """
        self._clear_state()
        self._randomize_walls(self.n_wall)
        self._add_lava(self.n_lava)

    def _randomize_walls(self, n_wall: int) -> None:
        """
            adds n_wall tiles to the game board in the form of a tree
            all changes are made to self._game_state
            n_wall: int, the number of tiles to add
        """
        wall_tree = self._get_all_tiles_of_type(required_type="Wall")

        def add_wall(point: np.ndarray) -> None:
            wall_tree.append(point)
            self._game_state[point[0], point[1], 0] = 1.

        for i in range(self.random_wall):
            point: np.ndarray = self._random_point("Empty")
            add_wall(point)
        for i in range(self.n_wall):
            point: np.ndarray = self._extend_tree(wall_tree, "Empty")
            if point is None:
                print(f"Added {i} walls in total")
                return
            add_wall(point)

    def _random_point(self, required_type: str = None) -> np.ndarray:
        """
            required_type : str, in Gridworld.tile_types or None
                will return a point of that type. If None, will 
                place no stipulations
            result : np.ndarray[int] : [2,] (y, x)
        """
        point: np.ndarray = np.asarray([np.random.randint(self.y_dim), np.random.randint(self.x_dim)])
        if required_type is not None and required_type != self._get_tile_type(point):
            return self._random_point(required_type)
        return point


    def _clip_point(self, point: np.ndarray) -> np.ndarray:
        """
            point: np.ndarray[int] : [2,], tile to get neighbors for, (y, x)
            returns : np.ndarray[int], all of form (y, x)
        """
        return np.clip(point, [0, 0], [self.y_dim+1, self.x_dim+1])
        
    def _get_neighboring_tiles(self, point: np.ndarray, include_diagonals:bool=False) -> List[np.ndarray]:
        """
            point: np.ndarray[int] : [2,], tile to get neighbors for, (y, x)
            include_diagonals: whether or not to include diagonals in the list 
            returns : List[np.ndarray[int]], all of form (y, x)
        """
        def is_diagonal(dx: int, dy: int):
            return (dx == 0) == (dy == 0)
        result = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if include_diagonals or (not is_diagonal(dx, dy)):
                    # no diagonals, exactly one must be 0
                    new_point = point + np.asarray([dy, dx], dtype=int)
                    new_point = self._clip_point(new_point)
                    if not np.array_equal(new_point, point):
                        # don't add self as neighbor
                        result.append(new_point)
        return array_unique(result)

    def _get_tile_type(self, point: np.ndarray) -> str:
        """
            point: np.ndarray[int] : [2,], tile to get neighbors for, (y, x)
            returns : str, in [None, "Empty", "Wall", "Lava", ]
            None indicates state not set, shouldn't occur?
        """
        assert np.array_equal(point, self._clip_point(point)) 
        # assert that point is valid within game bounds
        state: np.ndarray = self._game_state[point[0], point[1], :]
        # state: np.ndarray[float] : [3,]
        if state[0] == 1.:
            return "Wall"
        elif state[1] == 1.:
            return "Lava"
        else:
            return "Empty"

    def _get_all_tiles_of_type(self, required_type: str) -> List[np.ndarray]:
        """
            required_type: str, either None or in self.tile_types
            return: list of points, each np.ndarray[int] : [2,]
        """
        if required_type is None:
            return [np.asarray([y, x]) 
                for y in range(self._game_state.shape[0])
                for x in range(self._game_state.shape[1])]
        else:
            return list(filter(
                lambda el : self._get_tile_type(el) == required_type,
                self._get_all_tiles_of_type(required_type=None)))