from typing import TypeVar, List, Tuple, NamedTuple

State = TypeVar("State")
Goal = TypeVar("Goal")
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
    ["subgoal_trajectory", Trajectory],
    ["subgoal", Goal],
    ["goal_trajectory", Trajectory],
    ["goal", Goal],
    ["terminal_state", State]])

Environment = TypeVar("Environment")