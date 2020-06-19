from typing import Union, Dict, List

from numpy.random import RandomState 

from agent.memory import IMemory
from data_structures.trees import Node, Tree 
from misc.typevars import State, Action, Reward, Option, OptionData
from misc.typevars import Trajectory, Environment, Transition, TrainSample
from misc.utils import optional_random, bool_random_choice

class CompleteMemory(IMemory[State, Action, Reward, OptionData]):
    def __init__(self, 
            max_length: int = None,
            random_seed: Union[int, RandomState] = None):
        """
            Parameters
            ----------
            max_length: int>0 = None
            random_seed: Union[int, RandomState] = None
        """
        assert max_length is None or max_length > 0
        self.max_length = max_length
        self.random_seed: RandomState = optional_random(random_seed)

        self._num_options: Dict[Node[Option[OptionData]], int] = { }
        self._num_transitions: Dict[Node[Option[OptionData]], int] = { }
        self._trajectory_for: Dict[
            Node[Option[OptionData]], 
            Trajectory[State, Action, Reward]] = { }
        self.list_roots: List[Node[Option]] = [ ]

        self.current_option: Node[Option] = None

    def reset(self,
            env: Environment[State, Action, Reward],
            root_option: Node[Option[OptionData]],
            random_seed: Union[int, RandomState] = None) -> None:
        """
            Parameters
            ----------
            env: Environment
            root_option: Node[Option]
            random_seed: Union[int, RandomState] = None
        """
        if random_seed is not None:
            self.random = optional_random(random_seed)
        assert root_option not in self._num_options
        # check that we haven't seen this before
        if len(self.list_roots) >= self.max_length:
            to_remove: Node[Option[OptionData]] = self.list_roots.pop()
            del self._num_options[to_remove]
            del self._num_transitions[to_remove]
        self.list_roots.append(root_option)
        self._num_options[root_option] = 0
        self._num_transitions[root_option] = 0
        self._trajectory_for[root_option] = [ ]
        
        self.current_option = root_option

    def set_actionable_option(self, option_node: Node[Option[OptionData]]) -> None:
        """
            Parameters
            ----------
            option_node: Node[Option]
        """
        self.current_option = option_node

    def add_suboption(self, 
            new_node: Node[Option[OptionData]], 
            parent_node: Node[Option[OptionData]]) -> None:
        """
            Parameters
            ----------
            new_node: Node[Option]
            parent_node: Node[Option]
        """
        root_node: Node[Option[OptionData]] = Tree.get_root(parent_node) 
        self._num_options[root_node] += 1
        self._trajectory_for[new_node] = [ ]

    def view(self, 
            transition: Transition[State, Action, Reward]) -> None:
        """
            Parameters
            ----------
            transition: Transition
        """
        assert self.current_option in self._trajectory_for
        root_node: Node[Option[OptionData]] = Tree.get_root(self.current_option)
        # ^ could be cached
        self._num_transitions[root_node] += 1
        self._trajectory_for[self.current_option].append(transition)
        
    def trajectory_for(self, 
            option_node: Node[Option[OptionData]]) -> Trajectory:
        """
            Parameters
            ----------
            option_node: Node[Option]
            Returns
            -------
            trajectory: Trajctory
        """
        start_depth: int = option_node.depth 
        result: Trajectory[State, Action, Reward] = [ ]
        current_node: Node[Option[OptionData]] = option_node
        while current_node is not None and current_node.depth >= start_depth:
            result = self._trajectory_for[current_node] + result
            current_node = Tree.get_next_left(current_node)
        return result

    def initial_state_for(self, 
            option_node: Node[Option[OptionData]]) -> State:
        """
            Parameters
            ----------
            option_node: Node[Option]
            Returns
            -------
            state: State
        """
        trajectory: Trajectory[State, Action, Reward] = self._trajectory_for(option_node)
        if len(trajectory) > 0:
            return trajectory[0].state
        else:
            return self.terminal_state_for(option_node)

    def terminal_state_for(self, option_node: Node[Option[OptionData]]) -> State:
        """
            Parameters
            ----------
            option_node: Node[Option]
            Returns
            -------
            state: State
        """
        next_right: Node[Option[OptionData]] = Tree.get_next_right(goal_node)
        if next_right is None:
            return None 
        else:
            return self.initial_state_for(next_right)

    def sample(self, 
            num_samples: int = 1) -> List[TrainSample[State, Action, Reward, OptionData]]:
        """
            Parameters
            ----------
            num_samples: int = 1
            Returns
            -------
            samples: List[TrainSample] : [num_samples, ]
        """
        result: List[TrainSample[State, Action, Reward, OptionData]] = [ ]
        for _ in range(num_samples):
            episode_root_node: Node[Option[OptionData]] = bool_random_choice(
                self.list_roots, 
                self.random)
            parent_option_node: Node[Option[OptionData]] = bool_random_choice(
                Tree.list_nodes(episode_root_node),
                self.random)
            child_option_node: Node[Option[OptionData]] = bool_random_choice(
                Tree.list_nodes(parent_option_node),
                self.random)
            
            suboption_trajectory: Trajectory[State, Action, Reward, OptionData] = \
                self._trajectory_for(child_option_node)
            option_trajectory: Trajectory[State, Action, Reward, OptionData] = \
                self._trajectory_for(parent_option_node)

            initial_state: State = self.initial_state_for(child_option_node)
            midpoint_state: State = self.terminal_state_for(child_option_node)
            terminal_state: State = self.terminal_state_for(parent_option_node)

            result.append(TrainSample(
                initial_state,
                suboption_trajectory,
                child_option_node,
                midpoint_state,
                option_trajectory,
                parent_option_node,
                terminal_state))
        return result
            