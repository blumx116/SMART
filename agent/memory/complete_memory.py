from typing import Dict, NamedTuple, Union, List

from interface import implements
import numpy as np
from numpy.random import RandomState

from .i_memory import IMemory
from .observations import CompleteObservation 
from misc.typevars import State, Action, Reward, Goal, Trajectory, Transition
from misc.typevars import Trajectory, TrainSample, Environment
from misc.utils import NumPyDict, flatmap, optional_random, array_random_choice
from agent.memory.trees import Node, Tree


class EpisodeMemory(IMemory[State, Action, Reward, Goal]):
    def __init__(self, rand_seed: Union[int, RandomState] = None):
        """
            Parameters
            ----------
            rand_seed: Union[int, RandomState] = None
            Attributes
            ----------
            random: RandomState
                random seed to be used for random sampling
            trajectory_for: Dict[Node[Goal], Trajectory]
                chronological list of transitions observed while pursuing the goal at the value
                as the actionable goal
            current_goal: Node[Goal]
                the current actionable goal - the goal where all new transitions will be recorded
            num_obs: int
                the total number of observations/transitions currently in this memory
            cached_goal_list: List[Node[Goal]]
                ordered list of all goals in memory
            cached_super_goal_list: List[Node[Goal]]
                ordered list of all goals with subgoals in memory
            cached_sub_goal_list: List[Node[Goal]]
                ordered list of all goals that have parents in memory
        """
        self.random: RandomState = optional_random(rand_seed)
        self.trajectory_for: Dict[Node[Goa], Trajectory] = { } 
        self.current_goal: Node[Goal] = None
        self.num_obs: int = 0
        self.cached_goal_list: List[Node[Goal]] = [ ] 
        self.cached_super_goal_list: List[Node[Goal]] = [ ] 
        self.cached_sub_goal_list: List[Node[Goal]] = [ ]

    def view(self, state: State, action: Action, reward: Reward) -> None:
        """
            Adds the observed transition to memory
            Parameters
            ----------
            state: State
                the state the we just transitioned AWAY from - s0 in (s0, a, r, s1)
            action: Action
                the action that was taken at the previous timestep
            reward: Reward
                the reward received as a result of (state, action)
            Returns
            -------
            None
        """
        assert self.current_goal is not None 
        self.trajectory_for[self.current_goal].append(Transition(state, action, reward))
        self.num_obs += 1 

    @property
    def num_goals(self):
        return len(self.trajectory_for)

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        self.current_goal = goal_node
        self._add_goal_node(goal_node)

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        self._add_goal_node(subgoal_node)

    def _add_goal_node(self, goal_node: Node[Goal]) -> None:
        if goal_node not in self.trajectory_for:
            self.trajectory_for[goal_node] = [ ]
            self._update_caches()



    def _update_caches(self) -> None:
        """
            updates the lists caching the goals in order. To be called whenever 
            a new goal is added.
            self.cached_goal_list: All goal nodes, chronological order
            self.cached_super_goal_list: All goal nodes that have a subgoal
            self.cached_sub_goal_list: All goal nodes that are a subgoal to another node
        """
        self.cached_goal_list: List[Node[Goal]] = \
            Tree.list_nodes(Tree.get_root(self.current_goal))
        self.cached_super_goal_list = list(filter(
            lambda node: node.has_relation('left') or node.has_relation('right'),
            self.cached_goal_list))
        self.cached_sub_goal_list = list(filter(
            lambda node: node.has_relation('parent'),
            self.cached_goal_list))

    def get_subgoals(self, goal_node: Node[Goal]) -> List[Node[Goal]]:
        """
            Returns a list of all subgoals pursued while pursuing goal_node
            NOTE: includes goal_node, in chronological order
            NOTE: could be static, is not for consistency with other methods
            Parameters
            ----------
            goal_node: Node[Goal]
                the node corresponding to the goal that we are looking for thhe subgoals of 
                within memory.
            Returns
            -------
            subgoals: List[Node[Goal]]
                list of all nodes associated with the goals pursued while pursuing goal_node
                always has a length of at least length 1
        """
        min_depth: int = goal_node.depth 
        cur_node: Node[Goal] = goal_node 
        result: List[Node[Goal]] = [ ] 
        while cur_node is not None and cur_node.depth >= min_depth:
            result = [cur_node] + result
            cur_node = Tree.get_next_left(cur_node)
        return result

    def get_initial_state(self, goal_node: Node[Goal]) -> State:
        """
            Given the memory associated with goal_node, searches for the the state where
            the goal started in memory.
            If the trajectory for the goal is non-empty, then this is the first
            state in the trajectory. Otherwise, it is the same as the terminal
            state
            NOTE: goal_node must be in the current episode's memory
            Parameters
            ----------
            goal_node: Node[Goal]
                the node corresponding to the goal that we are looking for the initial
                state of within memory.
            Returns
            -------
            initial_state: State
                the state where the goal started being executed. None in trivial case
        """
        trajectory: Trajectory = self.get_trajectory(goal_node)
        if len(trajectory) > 0:
            return trajectory[0].state
        else:
            return self.get_terminal_state(goal_node)

    def get_terminal_state(self, goal_node: Node[Goal]) -> State:
        """
            Get the final state for the goal, which is the same as the initial
            state of the next goal. Returns None if the environment terminated
                while pursuing this goal
            NOTE: goal_node must be in the current episode's memory
            Parameters
            ----------
            goal_node: Node[Goal]
                the node corresponding to the goal that we are looking for the terminal
                state of whithin memory. 
            Returns
            -------
            terminal_state: State
                the state whree the goal finished being executed. None in trivial case
        """
        next_right: Node[Goal] = Tree.get_next_left(goal_node)
        if next_right is None:
            return None 
        else:
            return self.get_initial_state(next_right)


    def get_trajectory(self, goal_node: Node[Goal]) -> Trajectory:
        """
            Returns a list of all transitions observed while pursuing 
            the goal corresponding to goal_node. Aggregates the trajectories
            of subgoals to achieve this
            NOTE: goal_node must be in memory
            Parameters
            ----------
            goal_node: Node[Goal]
                the node corresponding to the goal that were are querying the terminal 
                state of. 
            Returns
            -------
            trajectory: Trajectory
                the trajectory observed while pursuing goal_node and associated subgoals.
                empty list if the goal was abandoned immediately
        """
        subgoals: List[Node[Goal]] = self.get_subgoals(goal_node)
        trajectories: List[Trajectory] = list(map(
            lambda node: self.trajectory_for[node],
            subgoals))
        return flatmap(trajectories)


    def sample_batch(self, count: int = 1) -> List[TrainSample]:
        """
            Returns 'count' training samples from this episode. 
            Randomly selects goal from all goals and then a random subgoal for that goal.
            NOTE: a goal may be selected as its own subgoal, resulting in learning V(phi, phi) = 0
            as well as the identity.
            Parameters
            ----------
            count: int
                the number of training smaples to return
            Returns
            -------
            samples: List[TrainSamples]
                the randomly selected training samples
        """
        assert count >= 1
        result: List[TrainSample] = []
        for _ in range(count):
            goal: Node[Goal] = array_random_choice(self.cached_goal_list, self.random)
            subgoal: Node[Goal] = array_random_choice(self.get_subgoals(goal), self.random)
            subgoal_trajectory: Trajectory = self.get_trajectory(subgoal)
            goal_trajectory: Trajectory = self.get_trajectory(goal)[len(subgoal_trajectory):] 
            # portion of goal_trajectory that doesn't overlap with subgoal_trajectory
            result.append(TrainSample(
                initial_state=self.get_initial_state(goal),
                subgoal_trajectory=subgoal_trajectory,
                subgoal=subgoal,
                goal_trajectory=goal_trajectory,
                goal=goal,
                terminal_state=self.get_terminal_state(goal)))
        return result

class CompleteMemory(IMemory[State, Action, Reward, Goal]):
    def __init__(self, max_length: int = None, rand_seed: Union[int, RandomState] = None):
        """
            Parameters
            ----------
            max_length: int
                the maximum number of episodes to store in memory. Unbounded
                if None
            rand_seed: Union[int, RandomState, None]
                the random seed to be used for random samples
            Attributes
            ----------
            random: RandomState
                from rand_seed, used for random sampling of batches
            max_length: int
                the maximum number of episodes to store in memory. Unbounded if None
            history: List[EpisodeMemory]
                list of episodes that have already been completed and are still
                in memory. Sampled from for sample_batch
            historical_goals_length: int 
                the total number of goals across all episodes in history
            cur_goals_length: int
                the total number of goals in the current episode
            current_episode: EpisodeMemory
                the current episode that is still being added to 
        """
        self.random: RandomState = optional_random(rand_seed)
        self.max_length: int = max_length if max_length is not None else np.inf
        self.history: List[EpisodeMemory] = []
        # history should probably be a linked-list like structure
        self.historical_goals_length: int = 0
        self.cur_goals_length: int = 0
        self.current_episode: EpisodeMemory = None

    def reset(self, env: Environment, state: State, goal: Goal) -> None:
        """
            Starts a new EpisodeMemory to store all new goals and observations in.
            Pushes the previous EpisodeMemory in to the bounded size queue
            Parameters
            ----------
            env: Environment
                the environment that the next episode will be trained in
            state: State
                the initial state of the new environment (should be equivalent to 
                env.state(), provided as argument to allow for preprocessing)
            goal: Goal
                the terminal goal of the new environment (should be equivalent to
                env.goal(), provided as argument to allow for preprocessing)
            Returns
            -------
            None
        """
        if self.current_episode is not None and self.current_episode.num_obs > 0:
            # if the current episode was non-trivial, add it to our history
            self.historical_goals_length += self.current_episode.num_goals
            self.history.append(self.current_episode)
        while len(self.history) > self.max_length:
            # if we already have max history length, pop the first episode
            self.historical_goals_length -= self.history[0].num_goals
            assert self.cur_goals_length >= 0
            self.history = self.history[1:]
        self.current_episode = EpisodeMemory(rand_seed=self.random)

    def view(self, state: State, action: Action, reward: Reward) -> None:
        """
            Adds transition to the memory
            Parameters
            ----------
            state: State
                the state the we just transitioned AWAY from - s0 in (s0, a, r, s1)
            action: Action
                the action that was taken at the previous timestep
            reward: Reward
                the reward received as a result of (state, action)
            Returns
            -------
            None 
        """
        self.current_episode.view(state, action, reward)
        self.cur_goals_length = len(self.history[:-1])
        
    def sample_batch(self, count: int) -> List[TrainSample]:
        """
            NOTE: currently samples uniformly among episodes, then samples from
            goals uniformly from that episode. This results in goals in shorter 
            episodes being oversampled
            Parameters
            ----------
            count: int > 0
                the number of TrainSamples to return
            Returns
            -------
            samples: List[TrainSample]
                the training samples sampled from the dataset
        """
        if len(self.history) == 0:
            return None
        assert count >= 0 
        result: List[TrainSample] = [ ]
        for _ in range(count):
            episode: EpisodeMemory = self.random.choice(self.history)
            result += episode.sample_batch(1)
        return result

    def get_trajectory(self, goal_node: Node[Goal]) -> Trajectory:
        """
            Returns a list of all transitions observed while pursuing 
            the goal corresponding to goal_node. Aggregates the trajectories
            of subgoals to achieve this
            NOTE: goal_node must be in the current episode's memory
            Parameters
            ----------
            goal_node: Node[Goal]
                the node corresponding to the goal that were are 
                querying the terminal state of
            Returns
            -------
            trajectory: Trajectory
                the trajectory observed while pursuing goal_node and associated 
                subgoals. Empty list if the goal was abandoned immediately
        """
        assert goal_node in self.current_episode.cached_goal_list
        return self.current_episode.get_trajectory(goal_node)

    @property
    def num_goals(self) -> int:
        return self.cur_goals_length + self.current_episode.num_goals

    def _observe_set_current_goal(self, goal_node: Node[Goal]) -> None:
        self.current_episode._observe_set_current_goal(goal_node)

    def _observe_add_subgoal(self, subgoal_node: Node[Goal], existing_goal_node: Node[Goal]) -> None:
        self.current_episode._observe_add_subgoal(subgoal_node, existing_goal_node)