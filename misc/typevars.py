from typing import TypeVar, List, Tuple, NamedTuple

State = TypeVar("State")
Option = TypeVar("Option")
Reward = TypeVar("Reward")
Action = TypeVar("Action")
Memory = TypeVar("Memory")
Transition = NamedTuple("Transition", [
    ["state", State],
    ["action", Action],
    ["reward", Reward]])
Trajectory = List[Transition]
TrainSample = NamedTuple("TrainSample", [
    ["initial_state", State],
    ["suboption_trajectory", Trajectory],
    ["suboption", Option],
    ["midpoint_state", State]
    ["option_trajectory", Trajectory],
    ["option", Option],
    ["terminal_state", State]])

Environment = TypeVar("Environment")