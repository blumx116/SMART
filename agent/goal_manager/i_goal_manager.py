from typing import Generic, TypeVar

from interface import Interface

from agent.memory.trees import Tree, Node 
from misc.typevars import State, Goal, Reward, Environment, Action


class IGoalManager(Interface, Generic[State, Goal]):
    def observe(self, state: State, action: Action, reward: Reward) -> None:
        pass

    def reset(self, env: Environment, goal: Goal) -> None:
        pass

    def select_next_subgoal(self, state: State, goal_node: Node[Goal]) -> Goal:
        pass 

    def should_abandon(self, state: State, goal_node: Node[Goal]) -> bool:
        pass 

    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> bool:
        pass 

    def step(self) -> None:
        pass

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        pass 

    def _observe_abandon_goal(self, goal_node: Node[Goal]) -> None:
        pass 

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        pass