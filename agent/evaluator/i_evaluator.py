from abc import ABC, abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option, OptionData 

class IEvaluator(IOptimizable[State, Action, Reward, OptionData]):
    @abstractmethod
    def select(self, 
            state: State, 
            possibilities: List[Option[OptionData]], 
            parent: Option[OptionData]) -> Option[OptionData]:
        pass 