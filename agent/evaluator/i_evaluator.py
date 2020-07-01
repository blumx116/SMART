from abc import ABC, abstractmethod
from typing import Generic, List, Optional

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option, OptionData 

class IEvaluator(IOptimizable[State, Action, Reward, OptionData]):
    @abstractmethod
    def select(self, 
            state: State, 
            possibilities: List[Option[OptionData]],
            prev_option: Optional[Option[OptionData]],
            parent_option: Option[OptionData]) -> Option[OptionData]:
        pass 