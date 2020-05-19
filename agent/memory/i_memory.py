from typing import Generic

from interface import Interface

from misc.typevars import State, Action, Reward, Goal, Trajectory, Environment
from misc.typevars import TrainSample

class IMemory(Interface, Generic[State, Action, Reward, Goal]):
    def reset(self, env: Environment, state: State, goal: Goal) -> None:
        pass

    def view(self, state: State, action: Action, reward: Reward) -> None:
        pass

    def sample_batch(self, count: int) -> List[TrainSample]:
        pass
    
    def get_trajectory(self, goal_node: Node[Goal]) -> Trajectory:
        pass