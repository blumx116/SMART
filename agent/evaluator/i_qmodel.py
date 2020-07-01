from abc import abstractmethod
from typing import Generic, List, Tuple, Union, Any, Optional

from numpy.random import RandomState

from env import IEnvironment
from misc.typevars import State, Option, Reward, OptionData

Option = Option[OptionData]

class IQModel(Generic[State, Reward, OptionData]):
    TrainingDatum = Tuple[
        State, # initial state
        Optional[Option[OptionData]],  # previous option
        Option[OptionData],  # sub option
        Option[OptionData]]  # option

    @abstractmethod
    def forward(self, 
            state: State,
            prev_option: Optional[Option[OptionData]],
            suboption: Option[OptionData], 
            option: Option[OptionData]):
        pass 

    @abstractmethod
    def optimize(self, 
            inputs: List["IQModel.TrainingDatum"],
            targets: List[Reward]) -> None:
        pass

    @abstractmethod
    def reset(self,
            env: IEnvironment[State, Any, Reward],
            random_seed: Union[int, RandomState] = None):
        pass