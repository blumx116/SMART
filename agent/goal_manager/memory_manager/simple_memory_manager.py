from typing import Generic, Tuple, List, Dict

from interface import Interface

from .i_memory_manager import IMemoryManager
from agent.memory.trees import Node, Tree
from agent.memory.observations import SimpleMemory
from misc.typevars import State, Action, Reward, Goal, Environment

class SimpleMemoryManager(implements(IMemoryManager[State, Action, Reward, Goal])):
    def __init__(self, gamma: float):
        self.gamma: float = gamma
        self.current_root: Node[Goal] = None 
        self.current_goal: Node[Goal] = None 
        self.current_state: State = None 
        self.previous_episodes: List[Node[Goal]] = None 
        self.trajectory_lookup: Dict[Node[Goal], SimpleMemory] = { } 

    def __len__(self) -> int:
        return len(self.previous_episodes)

    def view(self, state: State, action: Action, reward: Reward) -> None:
        self.current_state = state

        self.trajectory_lookup[self.current_goal].view(
            reward, state, self.current_goal.goal)
    
    def reset(self, env: Environment, goal: Goal) -> None:
        if self.current_root is not None:
            # we just finished an episode
            self._finish_trajectory()
            self.previous_episodes.append

        self.current_root = None
        self.current_state = env.state()

    def generate_sample(self) -> Tuple[State, Goal, float]:
        pass 

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        pass

    def _observe_abandon_goal(self, goal_node: Node[Goal]) -> None:
        pass

    def _finish_trajectory(self) -> None:
        self.trajectory_lookup[self.current_goal].term_state = self.current_state

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        if self.current_root is None:
            self.current_root = goal_node
        else:
            self._finish_trajectory()

        self.current_goal: Node[Goal] = goal_node
        self.trajectory_lookup[goal_node] = SimpleMemory(
            gamma =self.gamma,
            goal=self.current_goal,
            init_state=self.current_state,
            rewards=float)
        