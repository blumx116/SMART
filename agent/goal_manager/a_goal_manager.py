from typing import List, Tuple, Union

from interface import implements
from numpy.random import RandomState
from scipy.stats import logistic
from torch import Tensor

from agent.goal_manager import IGoalManager
from agent.memory.trees import Node, Tree
from agent.goal_manager.evaluator import IEvaluator
from agent.goal_manager.generator import IGenerator
from agent.memory import IMemory
from misc.typevars import State, Action, Reward, Goal, Environment, Trajectory, TrainSample
from misc.utils import optional_random, bool_random_choice


class AGoalManager(IGoalManager[State, Goal]):
    def __init__(self, 
        evaluator: IEvaluator, 
        generator: IGenerator, 
        rand_seed: Union[int, RandomState]):
        """
            Parameters
            ----------
            evaluator: IEvaluator
                evaluator to be used for evaluating goals
            generator: IGenerator
                geneator to be used for generating possible goals
            rand_seed: Union[int, RandomState] = None
                random seed to be used for squishing probabilities to 
                boolean random choices
                NOTE: generator and evaluator may have their own random seeds
            Attributes
            ----------
            evaluator: IEvaluator
                evaluator to be used for evaluating goals
            generator: IGenerator
                geneator to be used for generating possible goals
            random: RandomState
                random seed to be used for squishing probabilities to 
                boolean random choices
                NOTE: generator and evaluator may have their own random seeds
        """
        self.evaluator: Evaluator = evaluator
        self.generator: Generator = generator
        self.random: RandomState = optional_random(rand_seed)

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
        pass

    def reset(self, env: Environment, state: State, goal: Goal) -> None:
        """
            To be called before the start of each new episode. Used to clear memory,
            update knowledge of environment parameters, etc.
            env: Environment
                the new environment that will be tested in
            goal: Goal
                the final goal that the agent is trying to achieve in the environment
        """
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
        
    def should_abandon(self, trajectory: Trajectory, state: State, goal_node: Node[Goal]) -> bool:
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

    def optimize(self, samples: List[TrainSample]) -> None:
        """
            Runs one step of the optimizer for all machine learning modules
        """
        self.evaluator.optimize(samples)
        self.generator.optimize(samples)

    def _sigmoid_sample(value: float, squish: bool = True) -> bool:
        """
            if squish, applies sigmoid on value to apply sigmoid function
            to squish it to [0, 1] range. The selects true with that probability
        """
        if squish:
            probability: float = logistic.cdf(value)
        else:
            probability: float = value
        return bool_random_choice(probability, rand_seed=self.random)