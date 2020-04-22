from typing import Union

import numpy as np
RandomState = np.random.RandomState

from env.mazeworld import MazeWorld
from utils import array_contains, array_random_choice, optional_random

Point = np.ndarray # np.ndarray[int]: [2,] (y, x)
Action = np.ndarray # np.ndarray[int]: [4,], onehot
State = np.ndarray # np.ndarray[float] : [y_dim, x_dim, 3]
Goal = np.ndarray # np.ndarray[int]: [y_dim, x_dim, 1]

class GreedyMazeAgent:
    def __init__(self, env: MazeWorld=None, random_seed: Union[int, RandomState] = None):
        self.env = env 
        self.reset(env, random_seed)


    def reset(self, env: MazeWorld=None, random_seed: Union[int, RandomState] = None):
        if self.env is None:
            assert env is not None 
        self.env = env
        self.eps = 0.03
        self.eps_max = 0.7
        self.history = []
        self.n_repeats = 0  
        self.random = optional_random(random_seed)

    def act(self, state: State, goal: Goal) -> Action:
        assert self.env is not None
        if self.random.uniform() < self.get_rand_chance():
            return self.get_random_action()
        distances = map(
            lambda action: self._distance(self.env._calculate_move(action), goal),
            self.env.actions)
        cur_distance = self._distance(self.env._location, goal)
        distance_deltas = list(map(
            lambda distance : distance - cur_distance,
            distances))
        if np.min(distance_deltas) >= 0: # no moves getting us closer to goal
            return self.get_random_action()
        return self.env.actions[np.argmin(distance_deltas)]

    def get_random_action(self):
        return array_random_choice(self.env.actions, self.random)

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
        return np.sum(np.abs(location - goal)) #manhattan distance
