from typing import List, Union

from numpy.random import RandomState

from . import IPlanningTerminator
from misc.typevars import State, Option, TrainSample, Environment

class DepthPlanningTerminator(IPlanningTerminator):
    def __init__(self, max_depth: int):
        self.max_depth: int = max_depth

    def termination_probability(self, 
            state: State, 
            option: Option) -> float:
        return float(option.depth >= self.max_depth)

    def optimize(self, 
            samples: List[TrainSample], 
            step:int = None) -> None:
        pass

    def reset(self, 
            env: Environment, 
            random_seed: Union[int, RandomState] = None) -> None:
        pass