from interface import implements
from torch import Tensor

from agent.goal_manager import IGoalManager
from agent.memory.trees import Node, Tree
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.goal_manager.memory_manager import IMemoryManager
from misc.typevars import State, Goal, Reward, Environment
from misc.typevars import MemoryManager, Evaluator, Generator

class AGoalManager(implements(IGoalManager[State, Goal])):
    def __init__(self, memory_manager: IMemoryManager, evaluator: IEvaluator, generator: IGenerator):
        self.memory_manager: MemoryManager = memory_manager
        self.evaluator: Evaluator = evaluator
        self.generator: Generator = generator

    def choose_subgoal(self, possible_subgoals: List[Goal], state: State, goal_node: Node[Goal]) -> Goal:
        return self.evaluator.choose_subgoal(possible_subgoals, state, goal_node)[0]

    def generate_subgoals(self, state: State, goal_node: Node[Goal]) -> List[Goal]:
        return self.generator.generate_subgoals(state, goal_node.value)

    def view(self, state: State, action: Action, reward: Reward) -> None:
        self.memory_manager.view(state, action, reward)

    def reset(self, env: Environment, goal: Goal) -> None:
        self.memory_manager.reset(env, goal)
        self.evaluator.reset(env, goal)
        self.generator.reset(env, goal)

    def select_next_subgoal(self, state: State, goal_node: Node[Goal]) -> Goal:
        possible_subgoals: List[Goal] = self.generate_subgoals(state, goal_node)
        # len = n_subgoals
        return self.choose_subgoal(possible_subgoals, state, goal_node)
        
    def should_abandon(self, state: State, goal_node: Node[Goal]) -> bool:
        pass
        
    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> bool:
        pass 

    def step(self) -> None:
        self.evaluator.step(self.memory_manager)
        self.generator.step(self.memory_manager)

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        self.memory_manager._observe_add_subgoal(subgoal_node, existing_goal_node)

    def _observe_abandon_goal(self, goal_node: Node[Goal]) -> None:
        self.memory_manager._observe_abandon_goal(goal_node)

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        self.memory_manager._observe_abandon_goal(goal_node)