from typing import List, Optional

import numpy as np


from agent.generator import IGenerator
from data_structures.trees import Tree
from env.minigrid.types import OneHotImg, Point, Action, Reward
from env.minigrid.wrappers import tile_type
from misc.typevars import Option

class SimpleMinigridGenerator(IGenerator[OneHotImg, Action, Reward, Point]):
    def generate(self,
            state: OneHotImg,
            prev_option: Optional[Option[Point]],
            parent_option: Option[Point]) -> List[Option[Point]]:
        xdim: int = state.shape[0]
        ydim: int = state.shape[0]
        depth: int = parent_option.depth
        if prev_option is not None:
            depth = max(prev_option.depth, depth)
        child_depth: int = depth + 1
        result: List[Option[Point]]
        # not quite right, but haven't figured out good solution
        for x in range(xdim):
            for y in range(ydim):
                point: Point = np.asarray([x, y], dtype=np.int8)
                if tile_type(point) in ['Empty', 'Goal']:
                    result.append(Option(point, child_depth))
        return result
