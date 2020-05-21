from typing import List

from interface import implements

from .i_generator import IGenerator
from env.mazeworld.mazeworld import Point
from agent.memory import IMemory
from misc.typevars import Environment, Goal, State

class SimpleGridworldGenerator(IGenerator):
    def __init__(self):
        self.env: Environment = None 

    def reset(self, env: Environment, goal: Goal) -> None:
        self.env = env 

    def step(self, memory_manager: IMemory) -> None:
        pass

    def generate_subgoals(self, state: State, goal: Goal) -> List[Goal]:
        possibilities: List[Point] = self.env._all_tiles_of_type("Empty")
        return list(map(env._point_to_grid, possibilities))