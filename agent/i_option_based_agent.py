from abc import ABC, abstractmethod
from typing import Generic, Union

from numpy.random import RandomState

from misc.typevars import State, Action, Reward, OptionData, Option
from misc.typevars import Environment, Transition

Environment = Environment[State, Action, Reward]
Option = Option[OptionData]

class IOptionBasedAgent(ABC, Generic[State, Action, Reward, OptionData]):
    @abstractmethod
    def reset(self, 
            env: Environment[State, Action, Reward],
            root_option: Option[OptionData],
            random_seed: Union[int, RandomState] = None) -> None:
        pass 

    @abstractmethod
    def view(self, 
            transition: Transition[State, Action, Reward]) -> None:
        pass 

    @abstractmethod
    def act(self, 
            state: State, 
            option: Option[OptionData]) -> Action:
        pass 

    @abstractmethod
    def optimize(self) -> None:
        pass 


