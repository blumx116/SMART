from abc import abstractmethod
from typing import Generic

from agent import IOptimizable
from misc.typevars import State, Option, Reward

class IQModel(Generic[State, Reward, Option]):
    @abstractmethod
    def forward(self, state: State, suboption: Option, option: Option):
        pass 

    @abstractmethod
    def optimize(self, inputs: List[State, Option, Option], targets: List[Reward]) -> None:
        pass 