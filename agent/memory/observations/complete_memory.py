from typing import TypeVar, Generic

import numpy as np

from agent.memory.observations import IMemory, SimpleMemory

Goal = TypeVar("Goal")
State = TypeVar("Goal")

class CompleteMemory(IMemory[Goal, State]):
    def __init__(self, gamma: float, goal: Goal, n_obs: int = 0):
        super().__init__(gamma, goal, n_obs)
        self.goals: List[Goal] = []
        self.rewards: List[float] = []
        self.states: List[State] = [] 

    def observe(self, reward: float, state: State, goal: Goal) -> None:
        self.rewards.append(reward)
        self.states.append(state)
        self.goals.append(goal)

    def terminal_state(self) -> State:
        if len(self.states) == 0:
            return None 
        return self.states[-1]

    def initial_state(self) -> State:
        if len(self.states) == 0:
            return None 
        return self.states[0]

    def total_rewards(self) -> float:
        discounts: np.ndarray[float] = np.arange(len(self.rewards))
        # np.ndarray[float] : (len[observation], )
        return np.sum(discounts * np.asarray(self.rewards))

    def to_simple(self) -> SimpleMemory:
        return SimpleMemory(
            gamma=gamma,
            goal=self.goal,
            state=self.initial_state(),
            rewards=self.total_rewards(),
            term_state=self.terminal_state(),
            n_obs=self.n_obs
        )

    def __add__(self, other: SimpleMemory[Goal, State]) -> SimpleMemory[Goal, State]:
        assert self.terminal_state() == other.initial_state()
        assert self.gamma == other.gamma 
        result: CompleteMemory[Goal, State] = \
            CompleteMemory(gamma = gamma, 
            goal=other.goal, 
            n_obs=self.n_obs + other.n_obs)
        result.goals = self.goals + other.goals 
        result.rewards = self.rewards + other.rewards
        result.states = self.states + other.states[1:]
        return result

    

