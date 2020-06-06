from abc import ABC,  abstractmethod
from typing import Union, Generic

from numpy.random import RandomState

from data_structures.trees import Node
from misc.typevars import Option, Transition, Trajectory, Environment

class IMemory(ABC, Generic[State, Action, Reward, Option]):
    @abstractmethod
    def reset(self, 
            env: Environment[State, Action, Reward, Option], 
            root_option: Node[Option], 
            random_seed: Union[int, RandomState] = None) -> None:
        pass

    @abstractmethod
    def set_actionable_option(self, option_node: Node[Option]) -> None:
        pass 

    @abstractmethod
    def add_suboption(self, new_node: Node[Option], parent_node: Node[Option]) -> None:
        pass 

    @abstractmethod
    def view(self, transition: Transition) -> None:
        pass

    @abstractmethod
    def trajectory_for(self, option_node: Node[Option]) -> Trajectory:
        pass