from abc import ABC, abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option

class IEvaluator(IOptimizable[State, Action, Reward, Option]):
    @abstractmethod
    def select(self, 
            state: State, 
            possibilities: List[Option], 
            parent: Option) -> Option:
        pass 