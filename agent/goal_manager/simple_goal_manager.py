from typing import Union, Callable

from numpy.random import RandomState

from .a_goal_manager import AGoalManager
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.memory.trees import Node, Tree
from misc.typevars import State, Goal , Trajectory
from misc.utils import array_equal

class SimpleGoalManager(AGoalManager):
    def __init__(self, 
            evaluator: IEvaluator, 
            generator: IGenerator,
            max_depth: int, 
            fulfils_goal: Callable[[State, Goal], bool] = None,
            rand_seed: Union[int, RandomState] = None):
        super().__init__(evaluator, generator, rand_seed)
        self.max_depth: int = max_depth
        if fulfils_goal is None:
            fulfils_goal = array_equal
        self.fulfils_goal: Callable[[State, Goal], bool] = fulfils_goal

    def goal_fulfilled(self, state: State, goal_node: Node[Goal]) -> bool:
        pass

    def should_abandon(self, trajectory: Trajectory, state: State, goal_node: Node[Goal]) -> bool:
        return self.fulfils_goal(state, goal_node.value)

    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> float:
        right_parent: Node[Goal] = goal_node
        max_parent_depth: int = right_parent.depth
        left_parent: Node[Goal] = Tree.get_next_left(right_parent)
        if left_parent is not None:
            max_parent_depth = max(left_parent.depth, max_parent_depth)
        target_depth: int = max_parent_depth + 1
        return float(max_parent_depth >= self.max_depth)
        