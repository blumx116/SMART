from typing import TypeVar, Generic

Goal = TypeVar("Goal")
State = TypeVar("Goal")

class IMemory(Generic[Goal, State]):
    def __init__(self, gamma: float, goal: Goal, n_obs: int = 0):
        self.gamma = gamma 
        self.goal = goal
        self.n_obs = n_obs

    def view(self, reward: float, state: State, goal: Goal) -> None:
        self.n_obs += 1
        pass

    def terminal_state(self) -> State:
        pass 

    def initial_state(self) -> State:
        pass 

    def total_rewards(self) -> float:
        pass 

    def __add__(self, other: IMemory[Goal, State]) -> IMemory[Goal, State]:
        pass