from interface import implements
import numpy as np

from .i_simple_agent import ISimpleAgent
from env.mazeworld import MazeWorld, State, Action, Reward, Point, Goal

InnerType = ISimpleAgent[MazeWorld, Point, Action, Reward, Point]

class Grid2PointWrapper(ISimpleAgent[MazeWorld, State, Action, Reward, Goal]):
    def __init__(self, inner: InnerType):
        self.inner: InnerType = inner
        self.env: MazeWorld = None

    def act(self, state: State, goal: Goal) -> Action:
        return self.inner.act(self._convert_state(state), self._convert_goal(goal))

    def reset(self, env: MazeWorld, state: State, goal: Goal) -> None:
        self.env = env 
        return self.inner.reset(env, self._convert_state(state), self._convert_goal(goal))

    def observe(self, state: State, action: Action, reward: Reward) -> None:
        return self.inner.observe(self._convert_state(state), action, reward)

    def step(self) -> None:
        self.inner.step()

    def _convert_state(self, state: State) -> Point:
        assert self.env is not None
        location_data: np.ndarray = state[:,:,2, np.newaxis]
        return self.env._grid_to_point(location_data).astype(int)

    def _convert_goal(self, goal: Goal) -> Point:
        return self.env._grid_to_point(goal).astype(int)