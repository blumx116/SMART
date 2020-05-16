from typing import 

from .a_goal_manager import AGoalManager
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.goal_manager.memory_manager import IMemoryManager
from agent.memory.trees import Node
from misc.typevars import State, Goal 
from misc.utils import array_equal

class SimpleGoalManager(AGoalManager):
    def __init__(self, 
            evaluator: IEvaluator, 
            generator: IGenerator,
            max_depth: int, 
            fulfils_goal: Callable[[State, Goal], bool] = None):
        super().__init__(evaluator, generator)
        self.max_depth: int = max_depth
        if fulfils_goal is None:
            fulfils_goal = array_equal
        self.fulfils_goal: Callable[[State, Goal], bool] = fulfils_goal

    def should_abandon(self, state: State, goal_node: Node[Goal]) -> bool:
        return False

    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> float:
        return float(goal_node.depth >= self.max_depth)
        