from typing import List

import numpy as np

from .a_goal_manager import AGoalManager
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.goal_manager.memory_manager import IMemoryManager
from agent.memory.trees import Node
from misc.typevars import State, Goal
from misc.utils import NumPyDict

class UtilitarianGoalManager(AGoalManager):
    def __init__(self, memory_manager: IMemoryManager, evaluator: IEvaluator,
        generator: IGenerator, abandon_tolerance_mult: float, plan_tolerance_mult: float):
        super().__init__(memory_manager, evaluator, generator)
        self.abandon_tolerance_mult: float = abandon_tolerance_mult
        self.plan_tolerance_mult: float = plan_tolerance_mult
        self.tolerable_reward: NumPyDict[Goal, Reward] = None

    def reset(self, env: Environment, goal: Goal) -> None:
        super().reset(env, goal)
        self.tolerable_reward: NumPyDict[Goal, Reward] = NumPyDict(dtype=float)

    def choose_subgoal(self, possible_subgoals: List[Goal], state: State, goal_node: Node[Goal]) ->  Goal:
        subgoal, scores = self.evaluator.choose_subgoal(possible_subgoals, state, goal_node) 
        # goal: Goal, scores: List[float]
        self.tolerable_reward[subgoal] = self.calculate_tolerance(scores)
        return subgoal

    def calculate_tolerance(self, scores: List[float]) -> Reward:
        sorted_scores: np.ndarray = np.sort(scores)
        # tacit assumption that all scores are unique
        # np.ndarray[float] : [generator.n_options, ]
        highest_score: float = sorted_scores[-1]
        second_highest: float = sorted_scores[-2]
        difference: float = highest_score - second_highest
        return highest_score - (self.abandon_tolerance_mult * difference)

    def should_abandon(self, state: State, goal_node: Node[Goal]) -> bool:
        expected_reward: float  = \
            self.evaluator.estimate_path_reward(state, goal_node)
        return expected_reward < self.tolerable_reward[goal_node.value]

    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> bool:
        subgoals: List[Goal] = self.generator.generate_subgoals(state, goal_node.value)
        scores: List[float] = self.evaluator.score_subgoals(subgoals, state, goal_node.value)
        probabilites: List[float] = self.evaluator.selection_probabilities(subgoals, scores, state, goal_node.value)
        expected_planned_score = np.dot(np.asarray(scores), np.asarray(probabilites))
        expected_unplanned_score = self.evaluator.estimate_path_reward(state, goal_node.value)
        return (expected_planned_score / expected_unplanned_score) > self.plan_tolerance_mult

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        super()._observe_set_current_goal(goal_node)
        self.current_goal = goal_node

    def view(self, state: State, action: Action, reward: Reward) -> None:
        super().view(state, action, reward)
        node: Node[Goal] = self.current_goal
        while node is not None:
            self.tolerable_reward[node.value] -= reward 
            node = node.get_relation('parent')
        
