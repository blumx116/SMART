from typing import Tuple, Union

import numpy as np

from utils import optional_random, array_random_choice

State = np.ndarray # np.ndarray[float] : [y_dim, x_dim, 3]
Goal = np.ndarray # np.ndarray[int]: [y_dim, x_dim, 1]
Point = np.ndarray # np.ndarray[int]: [2,] (y, x)
Action = np.ndarray # np.ndarray[int]: [4,], onehot

class MazeWorld:
    tile_types = ["Wall", "Lava", "Empty"]
    actions = [ np.asarray([1,0,0,0]),
                np.asarray([0,1,0,0]),
                np.asarray([0,0,1,0]),
                np.asarray([0,0,0,1])]

    def __init__(self, grid: np.ndarray):
        self._grid: np.ndarray = grid.astype(float) # np.ndarray[float] : [y_dim, x_dim, 2]
        self._location: Point = None #np.ndarray[float] : [y_dim, x_dim, 1]
        self._goal: Point = None #np.ndarray[float] : [y_dim, x_dim, 1]

        self.y_dim = self._grid.shape[0]
        self.x_dim = self._grid.shape[1]

    def state(self) -> State:
        assert self._location is not None 
        return self.stack((self._grid, self._point_to_grid(self._location)), axis=2)

    def reset(self, rand_seed: Union[int, np.random] = None) -> State:
        random = optional_random(rand_seed)
        self._location, self._goal = self._create_random_problem(random)
        self._goal_as_grid = self._point_to_grid(self._goal)
        return self.state()

    def _calculate_move(self, action: Action) -> Point:
        current: Position = self._location
        current += self._action_to_direction(action) #calculate delta
        current = self._clip_point(current) #stay within bounds
        if self._tile_type_matches(current, "Wall"):
            return self._location #can't move in to a wall
        else:
            return current #move successful

    def step(self, action: Action) -> Tuple[State, float, bool, Goal]:
        self._location = self._move(action)
        reward = -10 if self._tile_type_matches(self._location, "Lava") else -1
        done = np.array_equal(self._location, self._goal)
        return self.state(), reward, done, self._goal_as_grid
        
    def _action_to_direction(self, action: Action) -> Point:
        assert np.abs(np.sum(action) -1) < 1e-6 
        #should at least be probability distribution
        # [Up, Right, Down, Left]
        action: int = np.argmax(action)
        direction: Point = np.zeros((2,), dtype=int)
        direction[action % 2] = 1
        if action >= 2:
            direction *= -1
        return direction 

    def _clip_point(self, point: Point) -> Point:
        return np.clip(point, [0, 0], [self.y_dim+1, self.x_dim+1])

    def get_tile_type(self, point: Point) -> str:
        assert np.array_equal(point, self._clip_point(point))
        tile: np.ndarray = self._grid[point[0], point[1], :]
        # tile: np.ndarray[float] : [2,]
        if state[0]:
            return "Wall"
        elif state[1]:
            return "Lava"
        else:
            return "Empty"

    def _tile_type_matches(self, point: Point, required_type: str = None) -> bool:
        assert required_type is None or required_type in MazeWorld.tile_types
        return required_type is None or self.get_tile_type(point) == required_type

    def _all_tiles_of_type(self, required_type: str = None) -> List[Point]:
        points: List[Point] = [np.asarray([y, x], dtype=int)
            for y in range(self.y_dim)
            for x in range(self.x_dim)]
        return list(filter(
            lambda point: self.tile_type_matches(point, required_type),
            points))

    def _random_tile_of_type(self, required_type: str = None, rand_seed: Union[int, np.random] = None) -> Point:
        random: np.random = optional_random(rand_seed)
        possibilities: List[Point] = self._all_tiles_of_type(required_type)
        return array_random_choice(possibilities, random)


    def _point_to_grid(self, point: Point) -> np.ndarray: #float, [y_dim, x_dim, 2]
        result: np.ndarray = np.full((self.y_dim, self.x_dim), 0, dtype=float)
        result[point[0], point[1]] = 1.
        return result

    def _grid_to_point(self, grid: Goal) -> Point:
        assert np.sum(grid) == np.max(grid) == 1 #one-hot

    def _create_random_problem(self, rand_seed: Union[int, np.random] = None) -> Tuple[Point, Point]:
        random = optional_random(rand_seed)
        start = self._random_tile_of_type("Empty", random)
        end = self._random_tile_of_type("Empty", random)
        while np.array_equal(start, end):
            end = self._random_tile_of_type("Empty", random)
        return start, end