from typing import Generic, Tuple, List, Union

from interface import Interface
import numpy as np


from agent.memory.trees import Node
from misc.typevars import State, Action, Reward, Goal, Environment, TrainSample

class IEvaluator(Generic[State, Action, Reward, Goal]):
    def reset(self, env: Environment, goal: Goal) -> None:
        pass

    def estimate_path_reward(self, state: Union[State, Goal], prev_goal: Goal,
        goal: Goal) -> float:
        pass 

    def choose_subgoal(self, possible_subgoals: List[Goal], 
        state: State, goal_node: Node[Goal]) -> Tuple[Goal, np.ndarray]:
        # returns chosen goal and scores
        pass

    def score_subgoals(self, subgoals: List[Goal], state: State, goal: Goal) -> np.ndarray:
        pass 

    def selection_probabilities(self, subgoals: List[Goal], scores: np.ndarray, 
        state: State, goal: Goal) -> np.ndarray:
        pass 

    def step(self, samples: List[TrainSample], step: int = None) -> None:
        pass 