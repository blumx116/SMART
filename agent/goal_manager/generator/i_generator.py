from typing import Generic

from interface import Interface

from agent.goal_manager.memory_manager import IMemoryManager
from misc.typevars import State, Action, Reward, Goal, Environment

class IGenerator(Interface, Generic[State, Action, Reward, Goal]):
    def reset(self, env: Environment, goal: Goal) -> None:
        pass

    def generate_subgoals(self, state: State, goal: Goal) -> List[Goal]:
        pass 

    def step(self, memory_manager: IMemoryManager) -> None:
        pass 