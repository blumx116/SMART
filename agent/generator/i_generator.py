from abc import ABC, abstractmethod
from typing import Generic, List, Optional

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option, OptionData

class IGenerator(IOptimizable[State, Action, Reward, OptionData]):
    @abstractmethod
    def generate(self, 
            state: State,
            prev_option: Optional[Option[OptionData]],
            parent_option: Option[OptionData]) -> List[Option[OptionData]]:
        pass 