from abc import abstractmethod
from typing import Generic, List, Tuple

from agent import IOptimizable
from misc.typevars import State, Option, Reward, OptionData 

Option = Option[OptionData]

class IQModel(Generic[State, Reward, OptionData]):
    @abstractmethod
    def forward(self, 
            state: State, 
            suboption: Option[OptionData], 
            option: Option[OptionData]):
        pass 

    @abstractmethod
    def optimize(self, 
            inputs: List[Tuple[State, Option[OptionData], Option[OptionData]]], 
            targets: List[Reward]) -> None:
        pass