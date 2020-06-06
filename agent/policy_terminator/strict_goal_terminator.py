from . import IPolicyTerminator

class StrictGoalTerminator(IPolicyTerminator):
    def __init__(self, goal_achieved: Callable[[State, Option], float]):
        self.goal_achieved: Callable[[State, Option], float] = goal_achieved

    def reset(self,
            env: Environment,
            random_seed: Union[int, RandomState] = None) -> None:
        pass

    def optimize(self,
            samples: List[TrainSample],
            step: int = None) -> None:
        pass

    def termination_probability(self, 
            trajectory: Trajectory, 
            state: State, 
            option: Option) -> bool:
        return self.goal_achieved(state, option)