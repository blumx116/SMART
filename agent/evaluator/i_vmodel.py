from abc import abstractmethod
from typing import Generic

import torch

from agent import IOptimizable
from misc.typevars import State, Reward, Option

class IVModel(Generic[State, Reward, Option]):
    @abstractmethod
    def forward(self, state: State, option: Option) -> torch.Tensor:
        pass 

    @abstractmethod
    def optimize(self, inputs: List[Tuple[State, Option]], targets: List[Reward]) -> None:
        pass