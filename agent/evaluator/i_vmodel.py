from abc import abstractmethod
from typing import Generic, List, Tuple

import torch

from agent import IOptimizable
from misc.typevars import State, Reward, Option, OptionData

class IVModel(Generic[State, Reward, OptionData]):
    @abstractmethod
    def forward(self, 
            state: State, 
            option: Option[OptionData]) -> torch.Tensor:
        pass 

    @abstractmethod
    def optimize(self, 
            inputs: List[Tuple[State, Option[OptionData]]], 
            targets: List[Reward]) -> None:
        pass