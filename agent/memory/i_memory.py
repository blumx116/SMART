from abc import ABC,  abstractmethod
from typing import Union, Generic, List

from numpy.random import RandomState

from data_structures.trees import Node
from env import IEnvironment
from misc.typevars import State, Action, Reward, OptionData
from misc.typevars import Option, Transition, Trajectory, TrainSample

class IMemory(ABC, Generic[State, Action, Reward, OptionData]):
    @abstractmethod
    def reset(self, 
            env: IEnvironment[State, Action, Reward],
            root_option: Node[Option[OptionData]], 
            random_seed: Union[int, RandomState] = None) -> None:
        pass

    @abstractmethod
    def set_actionable_option(self, 
            option_node: Node[Option[OptionData]]) -> None:
        pass 

    @abstractmethod
    def add_suboption(self, 
            new_node: Node[Option[OptionData]], 
            parent_node: Node[Option[OptionData]]) -> None:
        pass 

    @abstractmethod
    def view(self, 
            transition: Transition[State, Action, Reward]) -> None:
        pass

    @abstractmethod
    def trajectory_for(self, 
            option_node: Node[Option[OptionData]]) -> Trajectory:
        pass

    @abstractmethod
    def sample(self, 
num_samples: int = 1) -> List[TrainSample[State, Action, Reward, OptionData]]:
        pass