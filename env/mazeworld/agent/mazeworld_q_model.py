import math
from itertools import chain
from typing import List, Tuple, Union, Iterable

import numpy as np 
from numpy.random import RandomState
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

from agent.evaluator import IQModel
from env.mazeworld import Point, MazeWorld, State, OptionData, Reward, Action
from misc.typevars import Option

class MazeworldQModel(IQModel[State, Reward, Option[OptionData]]):
    def __init__(self, xdim: int, ydim: int, settings: int):
        self.env : MazeWorld = None 
        self.input_dims: List[int] = (1, xdim, ydim, 4)
        self.device: torch.device = settings['device']

        self.loss: nn.modules.loss = nn.MSELoss()

        conv_channels1 = 20
        conv_size1 = 3
        maxpool_size1 = 2
        conv_channels2 = 15
        conv_size2 = 3

        conv1_x = xdim - (conv_size1 - 1)
        conv1_y = ydim - (conv_size1 - 1)
        maxpool1_x = math.floor(conv1_x / maxpool_size1)
        maxpool1_y = math.floor(conv1_y / maxpool_size1)
        conv2_x = maxpool1_x - (conv_size2 - 1)
        conv2_y = maxpool1_y - (conv_size2 - 1)
        hidden_input = conv2_x * conv2_y * conv_channels2

        self.conv_layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(5, conv_channels1, conv_size1),
            nn.MaxPool2d(maxpool_size1),
            nn.Conv2d(conv_channels1, conv_channels2, conv_size2),
            nn.Flatten(),
            nn.LayerNorm(hidden_input)).to(self.device)
        self.ff_layers: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_input + 1, 50),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(50),
            nn.Linear(50, 1)).to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3)

    def parameters(self) -> Iterable[Parameter]:
        return chain(self.conv_layers.parameters(), self.ff_layers.parameters())

    def reset(self, 
            env: MazeWorld,
            random_seed: Union[int, RandomState] = None) -> None:
        self.env = env

    def forward(self, 
            state: State, 
            suboption: Option[OptionData],
            option: Option[OptionData]) -> torch.Tensor:
        depth: int = option.depth 
        option_point: Point = option.value 
        option_grid: np.ndarray = self.env._point_to_grid(option_point) 
        # np.ndarray[float] : [ydim, xdim, 1]
        suboption_point: Point = option.value
        suboption_grid: np.ndarray = self.env._point_to_grid(option_point)
        # np.ndarray[float] : [ydim, xdim, 1]
        representation: np.ndarray = np.concatenate(
            (state, suboption_grid, option_grid), axis=-1)
        representation: torch.Tensor = torch.from_numpy(representation)
        representation: torch.Tensor = representation.float().to(self.device)
        # torch.Tensor[float, self.device] : [ydim, xdim, 5]
        representation: torch.Tensor = representation.unsqueeze(0).permute((0, 3, 1, 2))
        # torch.Tensor[float, self.device] : [1, 5, ydim, xdim]
        flattened: torch.Tensor = self.conv_layers(representation)
        # torch.Tensor[float, self.device] : [1, hidden_input]
        depth: torch.Tensor = torch.Tensor([depth]).float().to(self.device).reshape((1,1,))
        # torch.Tensor[float, self.device] : [1, 1]
        flattened: torch.Tensor = torch.cat((flattened, depth), dim=1)
        # torch.Tensor[float, self.device]: : [1, hidden_input + 1]
        return self.ff_layers(flattened).squeeze(0)

    def optimize(self, 
            inputs: List[Tuple[State, Option, Option]], 
            targets: List[Reward]) -> None:
        preds: List[torch.Tensor] = [] 
        for (state, suboption, option) in inputs:
            preds.append(self.forward(state, suboption, option))
        #preds: List[torch.Tensor[float, device]: [1,]] : [len(targets)]
        loss = self.loss(preds, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()