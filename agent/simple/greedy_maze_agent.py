import numpy as np

from env.mazeworld import MazeWorld
from utils import array_contains, array_random_choice

Point = np.ndarray # np.ndarray[int]: [2,] (y, x)
Action = np.ndarray # np.ndarray[int]: [4,], onehot
State = np.ndarray # np.ndarray[float] : [y_dim, x_dim, 3]
Goal = np.ndarray # np.ndarray[int]: [y_dim, x_dim, 1]

class GreedyMazeAgent:
    def __init__(self, env: MazeWorld=None):
        self.env = env 

    def reset(self, env: MazeWorld=None):
        if self.env is None:
            assert env is not None 
        self.env = env
        self.eps = 0.03
        self.eps_max = 0.7
        self.history = []
        self.n_repeats = 0  

    def act(self, state: State, goal: Goal) -> Action:
        if np.random.uniform() < self.get_rand_chance():
            return array_random_choice(self.env.actions)
        assert self.env is not None 
        distances = map(
            lambda action: self._distance(self.env._calculate_move(action), goal),
            self.env.actions)
        cur_distance = self._distance(self.env._location, goal)
        distance_deltas = list(map(
            lambda distance : distance - cur_distance,
            distances))
        if np.min(distance_deltas) >= 0: # no moves getting us closer to goal
            return None 
        return self.env.actions[np.argmin(distance_deltas)]

    def get_rand_chance(self):
        chance = self.eps * (self.n_repeats + 1)
        return np.clip(chance, 0, self.eps_max)

    def observe(self, reward: float, state: State, goal: Goal) -> None:
        point = self.env._location #this is cheaty, but temporary
        if array_contains(point, self.history):
            self.n_repeats += 1
        else:
            self.history.append(self.env._location) 

    def _distance(self, location: Point, goal: Goal) -> float:
        goal_point: Point = self.env._grid_to_point(goal)
        return np.sum(np.abs(location - goal_point)) #manhattan distance
