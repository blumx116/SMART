from typing import TypeVar, Generic

import numpy as np
from numpy.random import RandomState

from agent import IAgent
from agent.goal_manager import IGoalManager
from agent.memory import IMemory
from agent.memory.trees import Tree, Node
from misc.typevars import State, Goal, Reward, Action
from misc.typevars import Environment
from misc.utils import bool_random_choice, optional_random

# DEBUG
amax = lambda s: np.unravel_index(np.argmax(s[:,:,-1]), s[:,:,-1].shape)

class SMARTAgent(Generic[State, Goal, Action, Reward, Environment]):
    def __init__(self, 
            goal_manager: IGoalManager[State, Goal], 
            low_level_agent: IAgent,
            memory: IMemory) :
        """
            Parameters
            ----------
            goal_manager: IGoalManager[State, Goal]
                manages choosing, abandoning and planning goals
            low_level_agent: IAgent,
                manages turning goals in to actions
            memory: IMemory
                keeps a record of previous observations
            Attributes
            ----------
            goal_manager: IGoalManager[State, Goal]
                manages choosing, abandoning and planning goals
            low_level_agent: IAgent,
                manages turning goals in to actions
            memory: IMemory
                keeps a record of previous observations
            current_goal: Node[Goal]
                the current goal being pursued
            actionable_goal: Node[Goal] or None
                either current goal or None, depending on whether or not
                current goal is actionable
            terminal_goal: Node[Goal]
                the highest level goal for the agent to pursue
        """
        self.goal_manager: IGoalManager[State, Goal] = goal_manager
        self.low_level_agent = low_level_agent
        self.memory: IMemory = memory

        self.__current_goal: Node[Goal] = None 
        self.__actionable_goal: Node[Goal] = None
        self.__terminal_goal: Node[Goal] = None

    def reset(self, env: Environment, state: State, goal: Goal) -> None:
        self.goal_manager.reset(env, state, goal)
        self.low_level_agent.reset(env, state, goal)
        self.memory.reset(env, state, goal)

        print(f"reset: {amax(state)}=>{amax(goal)}")

        self._terminal_goal = Node(goal)
        self._current_goal = self._terminal_goal
        self._actionable_goal = None

    def act(self, state: State) -> Action:
        cur_goal: Node[Goal] = self._current_goal
        while self._should_abandon(state, cur_goal):
            cur_goal = self._abandon_goal(cur_goal)
        if not self._is_actionable(cur_goal):
            cur_goal = self._plan(state, cur_goal)
        print(f"act: subgoal={amax(cur_goal.value)}")
        return self.low_level_agent.act(state, cur_goal.value)

    def view(self, state: State, action: Action, reward: Reward) -> None:
        self.goal_manager.view(state, action, reward)
        self.low_level_agent.view(state, action, reward)
        self.memory.view(state, action, reward)

    def optimize(self, step: int = None) -> None:
        self.goal_manager.optimize(self.memory.sample_batch(10), step)
        self.low_level_agent.optimize()

    def _abandon_goal(self, goal_node: Node[Goal]) -> Node[Goal]:
        assert self._goal_equal(goal_node, self._current_goal)
        assert not self._goal_equal(goal_node, self._terminal_goal)

        self._announce_abandon_goal(goal_node)
        self._current_goal = Tree.get_next_right(goal_node)
        return self._current_goal

    def _add_subgoal(self, subgoal: Goal, existing_goal_node: Node[Goal]) -> Node[Goal]:
        subgoal_node: Node[Goal] = Tree.add_left(subgoal, existing_goal_node)
        self._announce_add_subgoal(subgoal_node, existing_goal_node)
        return subgoal_node

    def _goal_equal(self, goal1: Node[Goal], goal2: Node[Goal]) -> bool:
        return goal1 == goal2

    def _is_actionable(self, goal_node: Node[Goal]) -> bool:
        return self._goal_equal(goal_node, self._actionable_goal)

    def _plan(self, state: State, existing_goal: Node[Goal]) -> Node[Goal]:
        if self._should_terminate_planning(state, existing_goal):
            self._current_goal = existing_goal
            self._actionable_goal = existing_goal
            return self._actionable_goal
        subgoal: Goal = self.goal_manager.select_next_subgoal(state, existing_goal) 
        subgoal_node: Node[Goal] = self._add_subgoal(subgoal, existing_goal)
        return self._plan(state, subgoal_node)

    def _should_abandon(self, state: State, goal_node: Node[Goal]) -> bool:
        if self._goal_equal(goal_node, self._terminal_goal):
            return False # can't abandon terminal goal
        return self.goal_manager.should_abandon(
                trajectory=self.memory.get_trajectory(goal_node),
                state=state, goal_node=goal_node)

    def _should_terminate_planning(self, state: State, goal_node: Node[Goal]) -> None:
        return self.goal_manager.should_terminate_planning(state, goal_node)

    @property
    def _current_goal(self) -> Node[Goal]:
        return self.__current_goal

    @_current_goal.setter
    def _current_goal(self, value: Node[Goal]) -> None:
        self._announce_set_current_goal(value)
        self.__current_goal = value

    @property 
    def _actionable_goal(self) -> Node[Goal]:
        return self.__actionable_goal

    @_actionable_goal.setter
    def _actionable_goal(self, value: Node[Goal]) -> None:
        self._announce_set_actionable_goal(value)
        self.__actionable_goal = value

    @property
    def _terminal_goal(self) -> Node[Goal]:
        return self.__terminal_goal

    @_terminal_goal.setter
    def _terminal_goal(self, value: Node[Goal]) -> None:
        self.__terminal_goal = value 

    def _announce_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        toprint: Node[Goal] = subgoal_node
        print("========adding subgoal============")
        while toprint is not None:
            print(amax(toprint.value))
            toprint = Tree.get_next_right(toprint)
        print("===================================")
        self.memory._observe_add_subgoal(subgoal_node, existing_goal_node)
        self.goal_manager._observe_add_subgoal(subgoal_node, existing_goal_node)

    def _announce_abandon_goal(self, goal_node: Node[Goal]) -> None:
        self.goal_manager._observe_abandon_goal(goal_node)

    def _announce_set_current_goal(self, goal_node: Node[Goal]) -> None:
        self.memory._observe_set_current_goal(goal_node)
        self.goal_manager._observe_set_current_goal(goal_node)

    def _announce_set_actionable_goal(self, goal_node: Node[Goal]) -> None:
        pass