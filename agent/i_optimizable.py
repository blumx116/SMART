from abc import ABC, abstractmethod 
from typing import List, Generic, Union

from numpy.random import RandomState

from misc.typevars import State, Action, Reward, OptionData
from misc.typevars import Environment, TrainSample

class IOptimizable(ABC, Generic[State, Action, Reward, OptionData]):
    @abstractmethod
    def optimize(self, 
            samples: List[TrainSample[State, Action, Reward, OptionData]], 
            step: int = None) -> None:
        pass

    @abstractmethod
    def reset(self, 
            env: Environment[State, Action, Reward], 
            random_seed: Union[int, RandomState] = None) -> None:
        pass