from typing import Callable, Union, List

from numpy.random import RandomState

from . import IPolicyTerminator
from misc.typevars import State, Action, Reward, Option, OptionData
from misc.typevars import TrainSample, Trajectory, Environment



class StrictGoalTerminator(IPolicyTerminator[State, Action, Reward, OptionData]):
    def __init__(self, goal_achieved: Callable[[State, Option], float]):
        self.goal_achieved: Callable[[State, Option], float] = goal_achieved

    def reset(self,
            env: Environment[State, Action, Reward],
            random_seed: Union[int, RandomState] = None) -> None:
        pass

    def optimize(self,
            samples: List[TrainSample[State, Action, Reward, OptionData]],
            step: int = None) -> None:
        pass

    def termination_probability(self, 
            trajectory: Trajectory[State, Action, Reward], 
            state: State, 
            option: Option[OptionData]) -> bool:
        return self.goal_achieved(state, option)