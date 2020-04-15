from typing import TypeVar, Generic

from agent.memory.observations import IMemory 

Goal = TypeVar("Goal")
State = TypeVar("Goal")

class SimpleMemory(IMemory[Goal, State]):
    def __init__(self, gamma: float, goal: Goal, init_state: State, rewards: float, 
            term_state: None, n_obs: int = 0):
        super().__init__(gamma, goal, n_obs)
        self.init_state: State = init_state 
        if term_state is None:
            term_state = init_state
        self.term_state: State = term_state
        self.rewards: float = rewards

    @override
    def observe(self, reward: float, state: State, goal: Goal) -> None:
        super().observe(reward, state, goal)
        self.rewards += (gamma ** self.n_obs) * reward
        self.state = state 
        self.goal = goal 

    def terminal_state(self) -> State:
        return self.term_state

    def initial_state(self) -> State:
        return self.init_state

    def total_rewards(self) -> float:
        return self.rewards 

    def __add__(self, other: SimpleMemory[Goal, State]) -> SimpleMemory[Goal, State]:
        assert self.terminal_state() == other.initial_state()
        assert self.gamma == other.gamma 
        result: SimpleMemory[Goal, State] = \
            SimpleMemory(gamma = gamma, 
            goal=other.goal, 
            state=self.initial_state(), 
            rewards=self.total_rewards() + \
                ((self.gamma ** self.n_obs) * other.total_rewards()), 
            term_state=other.terminal_state(),
            n_obs=self.n_obs + other.n_obs)
        return result