from typing import Dict

from interface import implements

from .i_memory import IMemory
from .observations import CompleteObservation 
from misc.typevars import State, Action, Reward, Goal, Trajectory, Environment
from misc.utils import NumPyDict
from agent.memory.trees import Node, Tree

class CompleteMemory(implements(IMemory[State, Action, Reward, Goal])):
    def __init__(self, gamma: float):
        self.episodes = [] 
        self.observations : NumPyDict[Node[Goal], CompleteObservation] = NumPyDict()
        self.root_node: Node[Goal] = None
        self.cur_goal_node: Node[Goal] = None
        self.cur_state: State = None

    def reset(self, env: Environment, goal: Goal) -> None:
        if self.root_node is not None:
            self.episodes.append(self.root_node)
        self.root_node = None
        self.cur_goal_node = None
        self.cur_state = env.state()

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        if self.root_node is None:
            self.root_node = goal_node
        if goal_node not in self.observations:
            self.observations[goal_node] = CompleteObservation(self.cur_state, goal_node.value)
        self.cur_goal_node = goal_node

    def view(self, state: State, action: Action, reward: Reward) -> None:
        self.cur_state = state 
        self.observations[self.cur_goal_node].view(state, action, reward)

    def trajectory_of(self, goal_node: Node[Goal]) -> Trajectory:
        trajectory: Trajectory = self.observations[goal_node].trajectory
        depth: int = goal_node.depth 
        cur_node: Node[Goal] = goal_node
        while cur_node.depth >= depth:
            cur_node: Node[Goal] = Tree.get_next_left(cur_node)
            trajectory = cur_node.trajectory + trajectory
        return trajectory

    def initial_state_of(self, goal_node: Node[Goal]) -> State:
        initial_state: State = self.observations[goal_node].initial_state()
        depth: int = goal_node.depth 
        cur_node: Node[Goal] = goal_node
        while cur_node.depth >= depth:
            cur_node: Node[Goal] = Tree.get_next_left(cur_node)
            initial_state = cur_node.initial_state
        return initial_state
