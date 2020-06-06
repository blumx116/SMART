from abc import ABC, abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option

class IGenerator(IOptimizable[State, Action, Reward, Option]):
    @abstractmethod
    def generate(self, state: State, option: Option) -> List[Option]:
        pass 