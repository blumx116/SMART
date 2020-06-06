from typing import List

from agent.generator import IGenerator
from env.mazeworld.mazeworld import Point
from misc.typevars import Environment, Option, State, TrainSample

class SimpleMazeworldGenerator(IGenerator[State, Action, Reward, Option]):
    def __init__(self):
        self.env: Environment = None 

    def reset(self, env: Environment) -> None:
        self.env = env 

    def optimize(self, samples: List[TrainSample], step: int = None) -> None:
        pass

    def generate(self, state: State, option: Option) -> List[Option]:
        possibilities: List[Point] = self.env._all_tiles_of_type("Empty")
        return list(map(
            lambda point: Option(point, option.depth+1),
            possibilities))