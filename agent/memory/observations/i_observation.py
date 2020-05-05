from typing import Generic

from interface import Interface

from misc.typevars import State, Action, Reward, Goal
class IObservation(Interface, Generic[State, Action, Reward, Goal]):
    def __init__(self, initial_state: State, goal: Goal):
        pass 

    def view(self, state: State, action: Action, reward: Reward) -> None:
        pass 

    def initial_state(self) -> State:
        pass 

    def terminal_state(self) -> State:
        pass 

    def total_reward(self, gamma: float) -> Reward:
        pass