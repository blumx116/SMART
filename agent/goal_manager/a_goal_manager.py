from typing import List, Tuple

from interface import implements
from torch import Tensor

from agent.goal_manager import IGoalManager
from agent.memory.trees import Node, Tree
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.goal_manager.memory_manager import IMemoryManager
from misc.typevars import State, Goal, Reward, Environment
from misc.typevars import MemoryManager, Evaluator, Generator

Trajectory = List[Tuple[State, Action, Reward]]]

class AGoalManager(implements(IGoalManager[State, Goal])):
    def __init__(self, evaluator: IEvaluator, generator: IGenerator):
        self.evaluator: Evaluator = evaluator
        self.generator: Generator = generator

    def choose_subgoal(self, possible_subgoals: List[Goal], state: State, goal_node: Node[Goal]) -> Goal:
        """
            Select which subgoal to pursue from the list of subgoals
            Parameters
            ----------
            possible_subgoals: List[Goal]
                the subgoals to choose from among
            state: State
                the state that the agent is at when making the decision
            goal_node: Node[Goal]
                the node containing the goal that the agent is currently trying to achieve
            Returns
            -------
            chosen: Goal
                the chosen goal that is recommended to pursue next
        """
        return self.evaluator.choose_subgoal(possible_subgoals, state, goal_node)[0]

    def generate_subgoals(self, state: State, goal_node: Node[Goal]) -> List[Goal]:
        """
            Generates the list of candidate subgoals to try to achieve the goal at 
            goal_node starting from state.
            NOTE: the list may be of any non-zero length
            Parameters
            ----------
            state: State
            goal_node: Node[Goal]
            Returns
            -------
            possible_subgoals: List[Goal]
        """
        return self.generator.generate_subgoals(state, goal_node.value)

    def view(self, state: State, action: Action, reward: Reward) -> None:
        """
            To be called once each time the agent observes a new state, action, reward
            tuple by interacting with the environment. Used to update state
            Parameters
            ----------
            state: State
            action: Action
            reward: Reward
        """
        self.memory_manager.view(state, action, reward)

    def reset(self, env: Environment, goal: Goal) -> None:
        """
            To be called before the start of each new episode. Used to clear memory,
            update knowledge of environment parameters, etc.
            env: Environment
                the new environment that will be tested in
            goal: Goal
                the final goal that the agent is trying to achieve in the environment
        """
        self.memory_manager.reset(env, goal)
        self.evaluator.reset(env, goal)
        self.generator.reset(env, goal)

    def select_next_subgoal(self, state: State, goal_node: Node[Goal]) -> Goal:
        """
            Recommends the next subgoal to be pursued in order to achieve the 
            goal_node starting from the state. 
            NOTE: this does not lock in the choice, only returns a selection
            Parameters
            ----------
            state: State
            goal_node: Node[Goal]
            Returns
            -------
            chosen: Goal
        """
        possible_subgoals: List[Goal] = self.generate_subgoals(state, goal_node)
        # len = n_subgoals
        return self.choose_subgoal(possible_subgoals, state, goal_node)
        
    def should_abandon(self, state: State, trajectory: Trajectory, goal_node: Node[Goal]) -> bool:
        """
            Returns the probability that the agent should stop pursuing its current goal.
            NOTE: it may be abandoned because it is no longer worthwhile, intractable,
            or because it has already been achieved
            Parameters
            ----------
            state: State
            goal_node: Node[Goal]
            Returns
            -------
            probability: float 
        """
        pass
        
    def should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> float:
        """
            Returns the probability that the agent should stop planning and immediately
            pursue its current goal.
            Parameters
            ----------
            state: State
            goal_node: Node[Goal]
            Returns
            -------
            probability: float
        """
        pass 

    def step(self) -> None:
        """
            Runs one step of the optimizer for all machine learning modules
        """
        self.evaluator.step(self.memory_manager)
        self.generator.step(self.memory_manager)

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        """
            Observes the addition of each subgoal. Used for 
        """
        self.memory_manager._observe_add_subgoal(subgoal_node, existing_goal_node)

    def _observe_abandon_goal(self, goal_node: Node[Goal]) -> None:
        self.memory_manager._observe_abandon_goal(goal_node)

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        self.memory_manager._observe_abandon_goal(goal_node)