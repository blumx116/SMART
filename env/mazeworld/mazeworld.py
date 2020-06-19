from typing import Tuple, Any


from gym import Space, spaces
import numpy as np

from env import i_environment
from . import MazeWorldCache, MazeWorldGenerator
from misc.typevars import State, Action, Reward

State = np.ndarray # np.ndarray[float] : [y_dim, x_dim, 3]
Point = np.ndarray # np.ndarray[int] : [2,] = (y, x)
Action = np.ndarray # np.ndarray[float] = [2,]
Reward = float

class Mazeworld(i_environment[State, Action, float]):
    def __init__(self,
            ydim: int,
            xdim: int,
            n_wall: int,
            n_lava: int,
            lava_cost: int,
            cache_dir: str):
        self.ydim: int = ydim
        self.xdim: int = xdim

    @property
    def action_space(self) -> Space:
        return spaces.Discrete(4)

    @property
    def observation_space(self) -> Space:
        return spaces.Box(0 ,1, shape=(self.xdim, self.ydim, 3))

    @property
    def reward_range(self) -> Tuple[float, float]:
        return (-10, -1)

    def close(self) -> None:
        pass

    def render(self, mode: str='human') -> Any:
        """

        :param mode: ['human', 'rgb_array', 'ansi']
        :return:
        """
        assert mode in ['human', 'rgb_array', 'ansi']
        if mode == 'ansi':
            return "" # not supported
        ...

    def reset(self) -> State:
        ...

    def seed(self) -> State:
        pass

    def step(self, action: Action) -> Tuple[State, Reward, bool, Any]:
        pass
