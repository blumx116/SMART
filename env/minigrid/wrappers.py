from dataclasses import dataclass
from typing import List, Dict

from gym import spaces, Space
from gym.core import ObservationWrapper
import numpy as np

from env import IEnvironment
from env.minigrid.types import OneHotImg, Action, Reward, RawState, DirectedPoint, Point
from misc.utils import np_onehot, is_onehot

"""
    my working knowledge of the RawState's encoding is as follows:
    
    RawState is an np.ndarray[uint8] : [x, y, 3]
    
    RawState[:,:,0] contains information about what objects are where
    0: Unseen
    1: Empty
    2: Wall
    3: Floor
    4: Door
    5: Key
    6: Ball
    7: Box
    8: Goal
    9: Lava
    10: Agent
    
    RawState[:,:,1] appears to be contain the color of each item
    Not 100% decoded yet
    
    RawState[:,:,2] encodes the direction of the objects
    0: Right (1, 0)
    1: Down (0, 1)
    2: Left (-1, 0)
    3: Up (0, -1)
    
"""

OneHotImg_dimensions: Dict[str, int] = {
    'Wall': 0,
    'Floor': 1,
    'Door': 2,
    'Key': 3,
    'Ball': 4,
    'Box': 5,
    'Goal': 6,
    'Lava': 7,
    'Agent': 8,
    'dx': 9,
    'dy': 10
}

OneHotImg_info_at_dim: List[str] = list(map(
    lambda kvpair: kvpair[0],
    sorted(
        OneHotImg_dimensions.items(),
        key=lambda kvpair: kvpair[1])))

class OnehotWrapper(ObservationWrapper, IEnvironment[OneHotImg, Action, Reward], ):
    def __init__(self,
                 inner: IEnvironment[RawState, Action, Reward]):
        super().__init__(inner)
        self.observation_space: Space = spaces.Box(low=0, high=1, shape=(self.height, self.width, 11))


    def observation(self, observation: RawState) -> OneHotImg:
        dim1: np.ndarray = np_onehot(observation[:, :, 0], max=10)
        useless_dims: List[int] = [0, 1]
        # 1 indicates empty, which is redundant
        # 0 indicates unseen, but we're working with perfect vision
        dim1 = np.delete(dim1, useless_dims, 2)
        # np.ndarray[int] : [xdim, ydim, 9]
        # all of the data in row 2 of input appear to be redundant

        angles: np.ndarray = (np.pi / 2) * observation[:, :, 2]
        dxs = np.cos(angles)[:, :, np.newaxis].astype(int)  # [x, y, 1] all {-1, 0, 1}
        dys = np.sin(angles)[:, :, np.newaxis].astype(int)  # same

        return np.concatenate((dim1, dxs, dys), axis=-1).astype(np.int8)

def find(observation: OneHotImg, obj_name: str) -> Point:
    location_dim: np.ndarray = observation[:, :, OneHotImg_dimensions[obj_name]]
    assert is_onehot(location_dim)
    point: Point = np.asarray(np.unravel_index(
        np.argmax(location_dim), location_dim.shape), dtype=np.int8)
    # np.ndarray[int] : [2, ]
    return point

def direction_at(observation: OneHotImg, point: Point) -> Point:
    dxdy_dims: List[int] = [OneHotImg_dimensions['dx'], OneHotImg_dimensions['dy']]
    direction_dims: np.ndarray = observation[:, :, dxdy_dims]
    #  np.ndarray[int]: [x, y, 2]
    direction: np.ndarray = direction_dims[tuple(point)]
    # np.ndarray[int]: [2, ]
    return direction

def tile_type(observation: OneHotImg, point: Point) -> str:
    relevant_dims: np.ndarray = observation[:, :, :8]
    # np.ndarray[x,y, 8]
    info_for_point: np.ndarray = relevant_dims[tuple(point)]
    # np.ndarray[8,]
    if np.max(info_for_point) == 0:
        return "Empty"
    else:
        info_code: int = np.argmax(info_for_point)
        assert np.sum(info_for_point == info_for_point[info_code]) == 1 # unique max
        return OneHotImg_info_at_dim[info_code]


# NOTE: make sure to change this whenever you change OnehotWrapper's encoding
def onehot2directedpoint(observation: OneHotImg) -> DirectedPoint:
    point: Point = find(observation, 'Agent')
    direction: np.ndarray = direction_at(observation, point)
    return np.concatenate((point, direction))
