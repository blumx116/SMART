from typing import List, Union, Optional

from numpy.random import RandomState

from . import IPlanningTerminator
from env import IEnvironment
from misc.typevars import State, Action, Reward, Option, TrainSample, OptionData

class DepthPlanningTerminator(IPlanningTerminator[State, Action, Reward, OptionData]):
    def __init__(self, max_depth: int):
        self.max_depth: int = max_depth

    def termination_probability(self, 
            state: State, 
            prev_option: Optional[Option[OptionData]],
            parent_option: Option[OptionData]) -> float:
        depth: int = max(
            map(lambda o: o.depth,
                filter(lambda o: o is not None, [prev_option, parent_option])))
        # max of depths of non-None options
        return float(depth >= self.max_depth)

    def optimize(self, 
            samples: List[TrainSample[State, Action, Reward, OptionData]], 
            step:int = None) -> None:
        pass

    def reset(self, 
            env: IEnvironment[State, Action, Reward],
            random_seed: Union[int, RandomState] = None) -> None:
        pass