from typing import Dict, Any, Callable, Optional

from numpy.random import RandomState

from agent.evaluator import AEvaluator, IVModel, IQModel
from agent.planning_terminator import IPlanningTerminator
from env.minigrid.types import State, Action, Reward, Point
from misc.typevars import Option


class Evaluator(AEvaluator[State, Action, Reward, Point]):
    def __init__(self,
            v_model: IVModel[State, Reward, Point],
            q_model: IQModel[State, Reward, Point],
            planning_terminator: IPlanningTerminator[State, Action, Reward, Point],
            settings: Dict[str, Any],
            get_beta: Callable[[int], float],
            gamma: float,):
        super().__init__(v_model, q_model, settings, get_beta, gamma)
        self.planning_terminator: IPlanningTerminator = planning_terminator
        self.random: RandomState = RandomState(settings['random'])

    def _should_use_raw_(self,
            state: State,
            prev_option: Optional[Option[Point]],
            option: Option[Point]) -> bool:
        proba: float = self.planning_terminator.termination_probability(state, prev_option, option)
        return self.random.random() < proba