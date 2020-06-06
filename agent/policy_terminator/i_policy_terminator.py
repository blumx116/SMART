from abc import abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option

class IPolicyTerminator(IOptimizable[State, Action, Reward, Option]):
    @abstractmethod
    def termination_probability(self, 
            trajectory: Trajectory, 
            state: State, 
            option: Option) -> float:
        pass 