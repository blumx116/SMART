from typing import List, Union

from numpy.random import RandomState

from . import IPlanningTerminator
from misc.typevars import State, Action, Reward, Option, TrainSample, Environment, OptionData

class DepthPlanningTerminator(IPlanningTerminator[State, Action, Reward, OptionData]):
    def __init__(self, max_depth: int):
        self.max_depth: int = max_depth

    def termination_probability(self, 
            state: State, 
            option: Option[OptionData]) -> float:
        return float(option.depth >= self.max_depth)

    def optimize(self, 
            samples: List[TrainSample[State, Action, Reward, OptionData]], 
            step:int = None) -> None:
        pass

    def reset(self, 
            env: Environment[State, Action, Reward], 
            random_seed: Union[int, RandomState] = None) -> None:
        pass