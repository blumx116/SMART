import numpy as np



class BasicMemory:
    def __init__(self, goal: Goal, states: List[State], rewards: np.ndarray[float]):
        self.goal: Goal = goal 
        self.states: List[State] = states
        self.rewards: np.ndarray

    def __add__(self, other: BasicMemory) -> BasicMemory: