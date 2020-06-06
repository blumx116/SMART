from abc import ABC, abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option

class IPlanningTerminator(IOptimizable[State, Action, Reward, Option]):
    @abstractmethod
    def termination_probability(self, state: State, option: Option) -> float:
        pass 