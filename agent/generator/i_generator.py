from abc import ABC, abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option, OptionData

class IGenerator(IOptimizable[State, Action, Reward, OptionData]):
    @abstractmethod
    def generate(self, 
            state: State, 
            option: Option[OptionData]) -> List[Option[OptionData]]:
        pass 