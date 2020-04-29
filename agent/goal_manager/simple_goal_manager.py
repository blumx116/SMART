

from .a_goal_manager import AGoalManager
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.goal_manager.memory_manager import IMemoryManager
from agent.memory.trees import Node
from misc.typevars import State, Goal 

class SimpleGoalManager(AGoalManager):
    def __init__(self, memory_manager: IMemoryManager, evaluator: IEvaluator, generator: IGenerator,
        max_depth: int):
        super().__init__(memory_manager, evaluator, generator)
        self.max_depth = max_depth

    def should_abandon(self, state: State, goal_node: Node[Goal]) -> bool:
        return False

    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> bool:
        return goal_node.depth >= self.max_depth
        