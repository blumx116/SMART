from typing import List, Union

from numpy.random import RandomState

from agent.generator import IGenerator
from env.mazeworld.mazeworld_old import State, Action, Reward, OptionData
from misc.typevars import Environment, Option, TrainSample

# note Option == Point

class SimpleMazeworldGenerator(IGenerator[State, Action, Reward, OptionData]):
    def __init__(self):
        self.env: Environment[State, Action, Reward] = None

    def reset(self,
            env: Environment[State, Action, Reward],
            random_seed: Union[int, RandomState] = None) -> None:
        self.env = env 

    def optimize(self,
            samples: List[TrainSample[State, Action, Reward, OptionData]], step: int = None) -> None:
        pass

    def generate(self,
            state: State,
            option: Option[OptionData]) -> List[Option[OptionData]]:
        possibilities: List[Option] = self.env._all_tiles_of_type("Empty")
        return list(map(
            lambda point: Option(point, option.depth+1),
            possibilities))