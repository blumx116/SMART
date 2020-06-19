from typing import Dict, Any, Callable

from agent.evaluator import AEvaluator, IVModel, IQModel
from agent.planning_terminator import IPlanningTerminator
from misc.typevars import State, Action, Reward, Option, OptionData
from misc.utils import bool_random_choice

class SimpleMazeworldEvaluator(AEvaluator[State, Action, Reward, OptionData]):
    def __init__(self,
        planning_terminator: IPlanningTerminator[State, Action, Reward, OptionData],
        v_model: IVModel[State, Reward, OptionData],
        q_model: IQModel[State, Reward, OptionData],
        settings: Dict[str, Any],
        get_beta: Callable[[int], float],
        gamma: float):

        self.planning_terminator: IPlanningTerminator[State, Action, Reward, OptionData] = \
            planning_terminator
        super().__init__(v_model, q_model, settings, get_beta, gamma)

    def _should_use_raw_(self, state, option):
        return bool_random_choice(
            self.planning_terminator.termination_probability(state, option), 
            self.random)