from typing import TypeVar, List, Tuple

State = TypeVar("State")
Goal = TypeVar("Goal")
Reward = TypeVar("Reward")
Action = TypeVar("Action")
Memory = TypeVar("Memory")
Trajectory = List[Tuple[State, Action, Reward]]

Environment = TypeVar("Environment")
MemoryManager = TypeVar("MemoryManager")
GoalBasedAgent = TypeVar("GoalBasedAgent")