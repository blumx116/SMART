from typing import Generic, Tuple, List

from interface import Interface

from agent.goal_manager.memory_manager import IMemoryManager
from misc.typevars import State, Action, Reward, Goal, Environment

class IEvaluator(Interface, Generic[State, Action, Reward, Goal]):
    def reset(self, env: Environment, goal: Goal) -> None:
        pass

    def estimate_path_reward(self, state: Union[State, Goal], 
        goal: Union[Goal, Node[Goal]]) -> float:
        pass 

    def choose_subgoal(self, possible_subgoals: List[Goal], 
        state: State, goal_node: Node[Goal]) -> Tuple[Goal, np.ndarray]:
        # returns chosen goal and scores
        pass

    def score_subgoals(self, subgoals: List[Goal], state: State, goal: Goal) -> np.ndarray:
        pass 

    def selection_proabilities(self, subgoals: List[Goal], scores: np.ndarray, 
        state: State, goal: Goal) -> np.ndarray:
        pass 

    def step(self, memory_manager: IMemoryManager) -> None:
        pass 