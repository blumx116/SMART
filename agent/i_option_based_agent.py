from abc import ABC, abstractmethod
from typing import Generic

from misc.typevars import State, Action, Reward, Option
from misc.typevars import Environment, Transition

Environment = Environment[State, Action, Reward]

class IOptionBasedAgent(ABC, Generic[State, Action, Reward, Option]):
    @abstractmethod
    def reset(self, env: Environment, root_option: Option) -> None:
        pass 

    @abstractmethod
    def view(self, transition: Transition) -> None:
        pass 

    @abstractmethod
    def act(self, state: State, option: Option) -> Action:
        pass 

    @abstractmethod
    def optimize(self) -> None:
        pass 


