from abc import ABC, abstractmethod 
from typing import List, Generic

from misc.typevars import State, Action, Reward, Option
from misc.typevars import Environment, TrainSample

class IOptimizable(ABC, Generic[State, Action, Reward, Option]):
    @abstractmethod
    def optimize(self, 
            samples: List[TrainSample], 
            step: int = None) -> None:
        pass

    @abstractmethod
    def reset(self, 
            env: Environment, 
            random_seed: Union[int, RandomState] = None) -> None:
        pass