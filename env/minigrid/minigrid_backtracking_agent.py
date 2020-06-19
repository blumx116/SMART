from typing import List, Optional, Union, Iterable

import numpy as np
from numpy.linalg import norm
from numpy.random import RandomState
from scipy.spatial.distance import cityblock as manhattan_distance

from agent import  IOptionBasedAgent
from data_structures import PriorityQueue, NumPyDict
from env import IEnvironment
from env.minigrid.types import Action, Reward, Point, DirectedPoint
from env.minigrid.wrappers import OneHotImg, \
    OneHotImg_dimensions, onehot2directedpoint, find, tile_type
from misc.typevars import Transition, Option
from misc.utils import optional_random, array_shuffle, is_onehot

class MinigridBacktrackingAgent(IOptionBasedAgent[OneHotImg, Action, Reward, Point]):
    directions: List[np.ndarray] = [
        np.asarray([1, 0], dtype=np.int8),
        np.asarray([-1, 0], dtype=np.int8),
        np.asarray([0, 1], dtype=np.int8),
        np.asarray([0, -1], dtype=np.int8)]

    def __init__(self):
        self.target_path: Optional[List[DirectedPoint]]= None
        self.waypoints: PriorityQueue[List[DirectedPoint]] = PriorityQueue()
        self.history: List[DirectedPoint] = []
        self.visited: NumPyDict[Point, bool] = NumPyDict(dtype=np.int8)
        self.current_goal: Optional[Point] = None
        self.random: RandomState = optional_random()
        self.backstep: int = 0  # used if we are in the process of backing up

    def reset(self,
            env: Optional[IEnvironment[OneHotImg, Action, Reward]],
            root_option: Option[Point],
            random_seed: Union[int, RandomState] = None) -> None:
        self.target_path = None
        self.waypoints = PriorityQueue()
        self.history = []
        self.visited = NumPyDict(dtype=np.int8)
        self.current_goal = root_option.value
        self.backstep = 0
        if random_seed is not None:
            self.random = optional_random(random_seed)

    def view(self,
            transition: Transition[OneHotImg, Action, Reward]) -> None:
        dpoint: DirectedPoint = onehot2directedpoint(transition.state)
        self._record_(dpoint)


    def act(self,
            state: OneHotImg,
            option: Option[Point]) -> Action:
        goal: Point = option.value
        if self.current_goal is None or not np.array_equal(goal, self.current_goal):
            self.reset(None, option, None)
        self._record_(onehot2directedpoint(state))
        self._add_potential_targets_(state, goal)
        self._set_target_path_(state, goal)
        if self.backstep != 0:
            # currently undoing forward action
            return self._backup_()
        return self._follow_target_path_(state)

    def optimize(self) -> None:
        pass

    @staticmethod
    def _achieved_goal_(
            state: OneHotImg,
            goal: Point) -> bool:
        location: Point = find(state, 'Agent')
        return np.array_equal(location, goal)

    def _add_potential_targets_(self,
            state: OneHotImg,
            goal: Point) -> None:
        location: Point = find(state, 'Agent')
        possibilities: Iterable[Point] = map(
            lambda dxdy: location + dxdy,
            MinigridBacktrackingAgent.directions)
        possibilities = filter(
            lambda point: self._is_valid_point_(state, point),
            possibilities)
        # ^ doesn't appear to do anything b/c minigrid has walls on the edges
        possibilities = filter(
            lambda point: tile_type(state, point) != "Wall",
            possibilities)
        possibilities = filter(
            lambda point: point not in self.visited,
            possibilities)
        for point in possibilities:
            path_to: List[DirectedPoint] = self._navigate_to_(
                From=onehot2directedpoint(state), To=point)
            distance: float = self._distance_(point, goal)
            self.waypoints.push(self._join_paths_(
                self.history,path_to),
                distance)

    def _backup_(self) -> Action:
        if self.backstep == 2:
            self._increment_backstep_()
            return 2  # forward
        else:
            self._increment_backstep_()
            return 1  # left

    def _distance_(self,
            point1: Point,
            point2: Point) -> float:
        return manhattan_distance(point1, point2)

    def _find_last_common_index_(self,
            series1: List[DirectedPoint],
            series2: List[DirectedPoint]) -> int:
        highest_index: int = -1
        for index in range(len(series1)):
            if index >= len(series2) or \
                    not np.array_equal(series1[index], series2[index]):
                return highest_index
            highest_index = index
        return highest_index

    def _follow_target_path_(self,
            state: OneHotImg) -> Action:
        last_common_index: int = self._find_last_common_index_(
            self.history, self.target_path)
        if last_common_index + 1 < len(self.history):
            #
            # history has things that target path doesn't
            return self._replicate_move_(
                From=self.history[-1],
                To=self.history[-2])
        else:
            return self._replicate_move_(
                From=self.target_path[last_common_index],
                To=self.target_path[last_common_index+1])

    def _increment_backstep_(self):
        self.backstep += 1
        self.backstep %= 5
        if self.backstep == 0:
            # remove all of the backing up from history
            self.history = self.history[:-5]

    def _is_valid_point_(self,
            state: OneHotImg,
            point: Point) -> bool:
        """
        Returns true if point is inbounds for xy values of the image
        :param state: used only for dimensions
        :param point: the point that we're checking, xy coordinates
        :return: if the xy coordinates specify a point on the image
        """
        return np.array_equal(
            np.clip(point, a_min=0, a_max=state.shape[:2]),
            point)

    @staticmethod
    def _join_paths_(
            first: List[DirectedPoint],
            second: List[DirectedPoint]) -> List[DirectedPoint]:
        result: List[DirectedPoint] = first
        for addition in second:
            if len(result) >= 2 and np.array_equal(addition, result[-2]):
                # we backtracked, so we can just delete the backtrack
                result = result[:-2]
            result = result + [addition]
        return result

    def _navigate_to_(self,
            From: DirectedPoint,
            To: Point) -> List[DirectedPoint]:
        from_point: Point = From[:2]
        from_dir: np.ndarary = From[2:]
        #np.ndarray[int8] : [2, ] [dx, dy]
        dxdy = To - from_point
        if np.all(dxdy == 0): # From = To
            return [] #already there, end trajectory
        assert is_onehot(abs(dxdy))
        # target is one distance from current point
        if np.dot(dxdy, from_dir) == -1:
            # we are facing the opposite direction of what we need
            new_dir = from_dir[::-1] # arbitrary 90 degree turn
            new_point = from_point
        elif np.array_equal(dxdy, from_dir):
            # already facing correct direction
            new_dir = from_dir
            new_point = from_point + from_dir #move in direction
        else: # need to turn 90 degrees to dxdy
            new_dir = dxdy
            new_point = from_point
        next_dir_point: DirectedPoint = np.concatenate(
            (new_point, new_dir)).astype(np.int8)
        return [next_dir_point] + self._navigate_to_(
            From=next_dir_point, To=To)

    def _record_(self,
            state: DirectedPoint) -> None:
        if len(self.history) == 0 or not np.array_equal(state, self.history[-1]):
            self.history = self._join_paths_(self.history, [state])
            self.visited[state[:2]] = True

    def _replicate_move_(self,
            From: DirectedPoint,
            To: DirectedPoint) -> Action:
        from_point: Point = From[:-2]
        to_point: Point = To[:-2]
        from_dir: Point = From[2:]
        to_dir: Point = To[2:]
        assert np.array_equal(from_point, to_point) or \
                np.array_equal(from_dir, to_dir)
        # can only turn OR move, not both
        assert np.dot(from_dir, to_dir) != -1  # can't replicate 180 degree turn
        assert is_onehot(abs(to_dir)) and is_onehot(abs(from_dir))
        if np.array_equal(from_point, to_point):
            sine: int = np.cross(from_dir, to_dir)
            if sine == 1:
                return 1  # right
            elif sine == -1:
                return 0  # left
            else:
                raise Exception("shouldn't get here")
        else:
            if np.array_equal((to_point - from_point), from_dir):
                # we are facing in the right direction
                return 2  # move forward
            else:
                # our last move was forward, so we need to rotate 180
                # then move forward,then rotate 180 to undo it
                return self._backup_()

    def _set_target_path_(self,
            state: OneHotImg,
            goal: Point) -> None:
        if self.target_path is None:
            self.target_path = self.waypoints.pop()
            return self._set_target_path_(state, goal)
        elif self.target_path[-1][:-2] in self.visited:
            self.target_path = None
            return self._set_target_path_(state, goal)
