from typing import TypeVar, List, Tuple

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

Environment = TypeVar("Environment")
MemoryManager = TypeVar("MemoryManager")
GoalBasedAgent = TypeVar("GoalBasedAgent")