from typing import Generic, List

from interface import Interface

from agent.memory import IMemory
from misc.typevars import State, Action, Reward, Goal, Environment
from misc.typevars import TrainSample

class IGenerator(Generic[State, Action, Reward, Goal]):
    def reset(self, env: Environment, goal: Goal) -> None:
        pass    

    def generate_subgoals(self, state: State, goal: Goal) -> List[Goal]:
        pass 

    def optimize(self, samples: List[TrainSample]) -> None:
        pass 