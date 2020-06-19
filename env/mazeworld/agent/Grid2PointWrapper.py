from typing import Union, List

from numpy.random import RandomState

from agent import IOptionBasedAgent
from env.mazeworld import State, Action, Point, Reward, OptionData
from misc.typevars import Option, Environment, Transition

class AgentGrid2PointStateWrapper(IOptionBasedAgent[State, Action, Reward, OptionData]):
    def __init__(self,
            inner: IOptionBasedAgent[Point, Action, Reward, OptionData]):
        self.inner: IOptionBasedAgent[Point, Action, Reward, OptionData] = inner
        self.env: Environment[State, Action, Reward] = None

    def reset(self,
            env: Environment[State, Action, Reward],
            root_option: Option[OptionData],
            random_seed: Union[int, RandomState]) -> None:
        self.env = EnvGrid2PointStateWrapper(env)
        return self.inner.reset(self.env, root_option, random_seed)

    def view(self,
            transition: Transition[State, Action, Reward]) -> None:
        transition: Transition[Point, Action, Reward] = Transition(
            self._grid_to_point(transition.state),
            transition.action,
            transition.reward)
        return self.inner.view(transition)

    def act(self,
            state: State,
            option: Option[OptionData]) -> Action:
        return self.inner.act(
            self._grid_to_point(state),
            option)

    def optimize(self) -> None:
        return self.inner.optimize()


"""
from interface import implements
import numpy as np

from agent import IOptionBasedAgent
from env.mazeworld import MazeWorld, State, Action, Reward, Point, Goal

InnerType = IAgent[MazeWorld, Point, Action, Reward, Point]

class Grid2PointWrapper(IAgent[MazeWorld, State, Action, Reward, Goal]):
    def __init__(self, inner: InnerType):
        self.inner: InnerType = inner
        self.env: MazeWorld = None

    def reset(self, env: MazeWorld, state: State, goal: Goal) -> None:
        self.env = env 
        return self.inner.reset(env, self._convert_state(state), self._convert_goal(goal))

    def act(self, state: State, goal: Goal) -> Action:
        return self.inner.act(self._convert_state(state), self._convert_goal(goal))

    def view(self, state: State, action: Action, reward: Reward) -> None:
        return self.inner.view(self._convert_state(state), action, reward)

    def optimize(self) -> None:
        self.inner.optimize()

    def _convert_state(self, state: State) -> Point:
        assert self.env is not None
        location_data: np.ndarray = state[:,:,2, np.newaxis]
        return self.env._grid_to_point(location_data).astype(int)

    def _convert_goal(self, goal: Goal) -> Point:
        return self.env._grid_to_point(goal).astype(int)
"""