from typing import TypeVar

import numpy as np

Goal = TypeVar("Goal")
State = TypeVar("State")

class CompleteMemory:
    def __init__(self, goal: Goal, states: List[State], )