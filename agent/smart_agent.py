from typing import Dict, Any, Union, Generic, List, Optional

import numpy as np 
from numpy.random import RandomState

from agent import IOptionBasedAgent
from agent.evaluator import IEvaluator
from agent.generator import IGenerator
from agent.planning_terminator import IPlanningTerminator
from agent.policy_terminator import IPolicyTerminator
from agent.memory import IMemory
from data_structures.trees import Tree, Node
from env import IEnvironment
from misc.typevars import State, Action, Reward, Option, OptionData
from misc.typevars import TrainSample, Transition, Trajectory
from misc.utils import optional_random, bool_random_choice

class SMARTAgent(Generic[State, Action, Reward, OptionData]):
    def __init__(self,
            evaluator: IEvaluator[State, Action, Reward, OptionData],
            generator: IGenerator[State, Action, Reward, OptionData],
            planning_terminator: IPlanningTerminator[State, Action, Reward, OptionData],
            policy_terminator: IPolicyTerminator[State, Action, Reward, OptionData],
            low_level: IOptionBasedAgent[State, Action, Reward, OptionData],
            memory: IMemory[State, Action, Reward, Option],
            settings: Dict[str, Any]):
        self.evaluator: IEvaluator[State, Action, Reward, OptionData] = evaluator
        self.generator: IGenerator[State, Action, Reward, OptionData] = generator
        self.planning_terminator: IPlanningTerminator[State, Action, Reward, OptionData] = \
            planning_terminator
        self.policy_terminator: IPolicyTerminator[State, Action, Reward, OptionData] = \
            policy_terminator
        self.low_level: IOptionBasedAgent[State, Action, Reward, OptionData] = low_level
        self.memory: IMemory[State, Action, Reward, OptionData] = memory 

        self.random: RandomState = optional_random(settings['random'])

        self.current_option_node: Node[Option[OptionData]] = None
        self.prev_option_node: Optional[Node[Option[OptionData]]] = None
        self.actionable_option_node: Node[Option[OptionData]] = None
        self.root_option_node: Node[Option[OptionData]] = None

    def optimize(self, step: int = None) -> None:
        samples: List[TrainSample[State, Action, Reward, OptionData]] = \
            self.memory.sample(50)
        self.evaluator.optimize(samples, step)
        self.generator.optimize(samples, step)
        self.planning_terminator.optimize(samples, step)
        self.policy_terminator.optimize(samples, step)
        self.low_level.optimize(step)

    def reset(self, 
            env: IEnvironment[State, Action, Reward],
            root_option: Option[OptionData], 
            random_seed: Union[int, RandomState] = None) -> None:
        """
            Reset the agent to function in a new environment/episode.
            Parameters
            ----------
            env: IEnvironment[State, Action, Reward]
                the environment the agent is about to act in
            root_option: Option
                the base option that the agent begins executing
        """
        self.root_option_node = Node(root_option)
        self.current_option_node = self.root_option_node
        self.prev_option_node = None
        if random_seed is not None:
            self.random = optional_random(random_seed)

        self.evaluator.reset(env, random_seed)
        self.generator.reset(env, random_seed)
        self.planning_terminator.reset(env, random_seed)
        self.policy_terminator.reset(env, random_seed)
        self.low_level.reset(env, root_option, random_seed)
        self.memory.reset(env, self.root_option_node, random_seed)

    def view(self, transition: Transition[State, Action, Reward]) -> None:
        """
            View a transition that happened in the environment, adding it to memory
            Parameters
            ----------
            transition: Transition | Tuple[State, Action, Reward, State]
                the transition that was observed
        """
        self.low_level.view(transition)
        self.memory.view(transition)

    def act(self, state: State, option: Option[OptionData] = None) -> Action:
        """
            Updates options internally, possibly abandoning its currrent
            options and choosing new ones. Finally chooses its option to 
            pursue and delegates the choice of choosing an action to the 
            lower level agent
            Parameters
            ----------
            state: State
                the observation of the state that the agent is at when making 
                the action
            option: Optional[Option]
                currently just for IOptionBasedAgent compatability
            Returns
            -------
            chosen: Action
                the action that the agent has chosen to take
        """
        while self._should_abandon_(state, self.current_option_node):
            self._abandon_option_(self.current_option_node)
        if not self._is_actionable_(self.current_option_node):
            self.plan(state, self.current_option_node)
            self._add_actionable_option_(self.current_option_node)
        return self.low_level.act(state, self.current_option_node.value)

    def plan(self, 
            state: State, 
            option_node: Node[Option[OptionData]]) -> Node[Option]:
        assert option_node == self.current_option_node
        # don't presently support forward planning
        if not self._should_stop_planning_(state, option_node):
            possibilities: List[Option[OptionData]] = \
                self.generator.generate(state,
                    prev_option=self._prev_option_(),
                    parent_option=option_node.value)
            chosen: Option[OptionData] = \
                self.evaluator.select(state, possibilities,
                    prev_option=self._prev_option_(),
                    parent_option=option_node.value)
            new_option_node: Node[Option[OptionData]] = \
                self._add_suboption_(chosen, option_node)
            return self.plan(state, new_option_node)
        else:
            return option_node

    def _should_abandon_(self, 
            state: State,
            option_node: Node[Option[OptionData]]) -> bool:
        trajectory: Trajectory[State, Action, Reward] = \
            self.memory.trajectory_for(option_node)
        return bool_random_choice(
            self.policy_terminator.termination_probability(
                trajectory, 
                state, 
                option_node.value),
            self.random)

    def _abandon_option_(self, 
            option_node: Node[Option[OptionData]]) -> None:
        assert not option_node == self.root_option_node
        if self._is_actionable_(option_node):
            self.actionable_option_node = None
        self.prev_option_node = self.current_option_node
        self.current_option_node = Tree.get_next_right(self.current_option_node)

    def _is_actionable_(self, 
            option_node: Node[Option[OptionData]]) -> bool:
        return self.actionable_option_node == option_node

    def _add_actionable_option_(self, 
            option_node: Node[Option[OptionData]]) -> None:
        self.actionable_option_node = option_node
        self.memory.set_actionable_option(option_node)

    def _add_suboption_(self,
            new_option: Option[OptionData], 
            parent_node: Node[Option[OptionData]]) -> Node[Option[OptionData]]:
        new_option_node: Node[Option[OptionData]] = \
            Tree.add_left(new_option, parent_node)
        self.memory.add_suboption(new_option_node, parent_node)
        self.current_option_node = new_option_node
        return new_option_node

    def _should_stop_planning_(self, 
            state: State, 
            option_node: Node[Option[OptionData]]) -> bool:
        assert option_node == self.current_option_node
        # don't presently support forward planning
        return bool_random_choice(
            self.planning_terminator.termination_probability(
                state,
                self._prev_option_(),
                option_node.value),
            self.random)

    def _prev_option_(self) -> Optional[Option[OptionData]]:
        if self.prev_option_node is None:
            return None
        return self.prev_option_node.value
