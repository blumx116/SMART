from typing import Generic

from interface import Interface

from misc.typevars import State, Action, Reward, Goal, Trajectory

class IMemory(Interface, Generic[State, Action, Reward, Goal]):
    def get_trajectory(self, state: State, goal_node: Node[Goal]) -> Trajectory:
        pass