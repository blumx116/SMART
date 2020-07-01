from abc import abstractmethod
from typing import Generic, List, Tuple, Union, Any, Optional

import torch
from numpy.random import RandomState

from env import IEnvironment
from misc.typevars import State, Reward, Option, OptionData


class IVModel(Generic[State, Reward, OptionData]):
    TrainingDatum = Tuple[
        State,  # initial state
        Optional[Option[OptionData]],  # prev option
        Option[OptionData]]  # option

    @abstractmethod
    def forward(self, 
            state: State,
            prev_option: Optional[Option[OptionData]],
            option: Option[OptionData]) -> torch.Tensor:
        pass 

    @abstractmethod
    def optimize(self, 
            inputs: List["IVModel.TrainingDatum"],
            targets: List[Reward]) -> None:
        pass


    @abstractmethod
    def reset(self,
              env: IEnvironment[State, Any, Reward],
              random_seed: Union[int, RandomState] = None):
        pass
