from typing import Generic

from interface import Interface

from misc.typevars import State, Action, Reward, Goal, Environment

class IMemoryManager(Interface, Generic[State, Action, Reward, Goal]):
    def reset(self, env: Environment, goal: Goal) -> None:
        pass

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        pass

    def _observe_abandon_goal(self, goal_node: Node[Goal]) -> None:
        pass

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        pass