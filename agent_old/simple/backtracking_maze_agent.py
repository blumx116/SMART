from typing import Set, List, NamedTuple, Iterable, Tuple

import numpy as np
from interface import implements

from agent import IAgent
from env.mazeworld import MazeWorld, Point, Action, Reward
from data_structures import PriorityQueue, NumPyDict

State = Point # np.ndarray[float] : [y_dim, x_dim, 3]
Goal = Point # np.ndarray[int]: [2,] (y, x)

#using this instead of NamedTuple for easy hashability
class Target:
    def __init__(self, point: Point, states: List[State]):
        self.point : Point = point 
        self.states: List[State]  = states 

# Target = NamedTuple("Target", [('point', Point), ('states', List[State])]) 

class BacktrackingMazeAgent(IAgent[MazeWorld, State, Action, Reward, Goal]):
    def __init__(self, env: MazeWorld):
        self.reset(env, None, None)

    def reset(self, env: MazeWorld, state: State, goal: Goal) -> None:
        self.env: MazeWorld = env 
        self.queue: PriorityQueue[Point] = PriorityQueue()
        self.visited: NumPyDict[Point, bool] = NumPyDict(int)
        self.visited[state] = True
        self.current_goal: Goal = goal
        self.history: List[State] = [state] 
        self.current_target: Target = None

    def act(self, state: State, goal: Goal) -> Action:
        """
            If we have a move that gets us closer to the goal to a point
            we haven't been before, take it. Otherwise, figure out the point
            that we haven't been to yet which is closest to the goal
            and start backtracking to that
        """
        if not array_equal(self.current_goal, goal):
            #forget everything whenever we switch goals
            self.reset(self.env, state, goal)
        if self.current_target is not None:
            if np.array_equal(self.current_target.point, state) or \
                self.current_target.point in self.visited:
                # abandon target if we reached it
                self.current_target = None 
        if self.current_target is None:
            # add all adjacent tiles as possibilities
            candidates: List[Tuple[int, Target]] = self._get_possible_targets(state, goal)
            for dist_to_goal, target in candidates:
                self.queue.put(target, dist_to_goal)
            #choose the point closest to the goal to pursue
            self.current_target: Target = self.queue.get()
            while self.current_target.point in self.visited:
                self.current_target = self.queue.get()
            return self.act(state, goal)
        else:
            return self._move_towards(self.current_target)

    def view(self, state: State, action: Action, reward: Reward) -> None:
        """
            Add the observed state to our history. If this move was a backtrack,
            then just remove the node we backtracked over instead so that we don't
            have any cycles
        """
        self.visited[state] = True 
        if len(self.history) > 1 and array_equal(self.history[-2], state):
                #our last move was a backtrack
                self.history = self.history[:-1] 
                #remove cycles from self.history
        else:
            self.history.append(state)

    def optimize(self) -> None:
        pass 

    def _are_opposite_actions(self, action1: Action, action2: Action) -> bool:
        """
            Checks whether or not 2 actions undo each other
        """
        dir1: Point = self.env._action_to_direction(action1)
        dir2: Point = self.env._action_to_direction(action2)
        return np.array_equal(dir1 + dir2, np.zeros((2,), dtype=int))
        #check if actions cancel each other out

    def _get_opposite_action(self, action: Action) -> Action:
        """
            Returns the action that would 'undo' the one provided as a parameter
        """
        return self.env._direction_to_action(
            -1 * (self.env._action_to_direction(action)))

    def _get_possible_targets(self, state: State, goal: Goal) -> List[Tuple[int, Target]]:
        """
            For each of the points that are adjacent to 'state' but haven't
            been visited yet, compiles the trajectory that would have been 
            taken to get to that point, then calculates the manhattan distance
            from that point to the goal
        """
        possibilities: Iterable[Point] = map(
            lambda action: self.env._calculate_move(action),
            self.env.actions)
        possibilities: Iterable[Target] = map(
            lambda point: Target(point=point, states=self.history + [point]),
            possibilities)
        possibilities: Iterable[Target] = filter(
            lambda target: target.point not in self.visited,
            possibilities)
        possibilities: Iterable[Tuple[int, Target]] = map(
            lambda target: (self._manhattan_distance(target.point, goal), target),
            possibilities)
        possibilities: Iterable[Tuple[int, Target]] = sorted(
            possibilities,
            key=lambda tup: tup[0])
        return list(possibilities)
    
    def _manhattan_distance(self, point: Point, goal: Goal) -> int:
        return int(np.sum(np.abs(point - goal)))

    def _index_of_last_shared_state(self, trajectory1: Iterable[State], trajectory2: Iterable[State]) -> State:
        """
            returns 'index' s.t. that trajectory1[:index+1] + trajectory2[:index+1] are
            identical -> in other words, the highest value such that the state at that 
            index and all indices before are the same between two trajectories.
        """
        index: int = -1
        for index, (state1, state2) in enumerate(zip(trajectory1, trajectory2)):
            if not array_equal(state1, state2):
                return index - 1
        return index

    def _move_towards(self, target: Target) -> Action:
        """
            calculates the move towards a target point. Note that the 
            target point must be a point that the agent was adjacent to at
            some point since receiving its most recent goal
        """
        cur_trajectory: List[State] = self.history
        target_trajectory: List[Action] = target.states
        index: int = self._index_of_last_shared_state(cur_trajectory, target_trajectory)
        if index == len(cur_trajectory) - 1:
            # cur_trajectory is a subset of target_trajectory
            assert len(target_trajectory) > len(cur_trajectory)
            return self._replicate_move(From=target_trajectory[index], To=target_trajectory[index+1])
        else:
            # cur_trajectory has diverged from target_trajectory, need to backtrack
            assert len(cur_trajectory) >= 2
            return self._replicate_move(From=cur_trajectory[-1], To=cur_trajectory[-2])

    def _replicate_move(self, From: State, To: State) -> Action:
        """
            Returns the move necessary to move from 'From' to 'To'
            'From' and 'To' need to be adjacent
        """
        direction: Point = To - From 
        assert np.sum(np.abs(direction)) == 1
        return self.env._direction_to_action(direction)