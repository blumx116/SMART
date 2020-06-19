from dataclasses import dataclass
from typing import TypeVar, List, Tuple, NamedTuple, NewType, Generic


State = TypeVar("State")
Reward = TypeVar("Reward")
Action = TypeVar("Action")
OptionData = TypeVar("OptionData")

@dataclass 
class Option(Generic[OptionData]):
    value: OptionData
    depth: int 

@dataclass
class Transition(Generic[State, Action, Reward]):
    state: State 
    action: Action
    reward: Reward 

class Trajectory(
    Generic[State, Action, Reward], 
    List[Transition[State, Action, Reward]]):
    pass 

@dataclass
class TrainSample(Generic[State, Action, Reward, OptionData]):
    initial_state: State
    suboption_trajectory: Trajectory[State, Action, Reward]
    suboption: Option 
    midpoint_state: State 
    option_trajectory: Trajectory [State, Action, Reward]
    option: Option 
    terminal_state: State

class Environment(Generic[State, Action, Reward]):
    pass 