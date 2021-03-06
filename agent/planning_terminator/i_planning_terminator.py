from abc import ABC, abstractmethod
from typing import Optional

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option, OptionData

class IPlanningTerminator(IOptimizable[State, Action, Reward, OptionData]):
    @abstractmethod
    def termination_probability(self, 
            state: State, 
            prev_option: Optional[Option[OptionData]],
            parent_option: Option[OptionData]) -> float:
        pass 