from itertools import count
from typing import Generic, TypeVar, List, Tuple

import heapq

T = TypeVar("T")

class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[int, T]] = []
        self._counter = count()

    def __len__(self) -> int:
        return len(self.elements)
    
    def empty(self) -> bool:
        return len(self.elements) == 0
    
    def put(self, item: T, priority: float) -> None:
        heapq.heappush(self.elements, (priority, -next(self._counter), item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[2]