from abc import abstractmethod, abstractproperty
from typing import Generic, Any, Tuple, Union

from gym import Space
from numpy.random import RandomState

from misc.typevars import State, Action, Reward

class IEnvironment(Generic[State, Action, Reward]):
    @abstractproperty
    def action_space(self) -> Space:
        pass

    @abstractproperty
    def observation_space(self) -> Space:
        pass

    @abstractproperty
    def reward_range(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def render(self, mode: str='human') -> Any:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def seed(self, seed: Union[int, RandomState]) -> None:
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, bool, Any]:
        pass

    def unwrapped(self) -> "IEnvironment":
        return self

    def __enter__(self) -> "IEnvironment":
        return self

    def __exit__(self) -> None:
        self.close()

    @abstractmethod
    def __str__(self) -> str:
        pass

