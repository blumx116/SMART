from abc import abstractmethod
from typing import Generic, List

from agent import IOptimizable
from misc.typevars import State, Action, Reward, Option, Trajectory, OptionData

class IPolicyTerminator(IOptimizable[State, Action, Reward, OptionData]):
    @abstractmethod
    def termination_probability(self, 
            trajectory: Trajectory[State, Action, Reward], 
            state: State, 
            option: Option[OptionData]) -> float:
        pass 