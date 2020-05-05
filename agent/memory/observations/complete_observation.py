from collections import namedtuple

from interface import implements

from .i_observation import IObservation
from misc.typevars import State, Action, Reward, Goal 

Transition = namedtuple("Transition", ["state", "action", "reward"])

class CompleteObservation:
    def __init__(self, initial_state: State, goal: Goal):
        self._initial_state = initial_state
        self.goal = goal 
        self.trajectory = []

    def view(self, state: State, action: Action, reward: Reward) -> None:
        self.trajectory.append((state, action, reward))

    def initial_state(self) -> State:
        return self._initial_state

    def terminal_state(self) -> State:
        if len(self.trajectory) == 0:
            return self._initial_state
        else:
            return self.trajectory[-1].state

    def total_reward(self, gamma: float) -> Reward:
        rewards = map(lambda trans: trans.reward, self.trajectory)
        sum: Reward  = 0 
        mult: float  = 1
        for reward in rewards:
            sum += mult * reward 
            mult *= gamma 
        return sum 