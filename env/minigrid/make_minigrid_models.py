import math
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

from agent.evaluator import IQModel, IVModel
from env import IEnvironment
from misc.typevars import State, Action, Reward, Option, OptionData
from misc.utils import Stacker

class EncoderModel(nn.Module):
    def __init__(self,
            image_shape: Tuple[int, int],
            n_features: int,
            num_inputs: int,
            device: torch.device):
        super().__init__()
        self.num_inputs: int = num_inputs
        self.image_shape: Tuple[int, int] = image_shape
        self.device: torch.device = device
        stacker = Stacker(input_shape=[-1, image_shape[0], image_shape[1], num_inputs])
        stacker.stack(nn.Conv2d(num_inputs, out_channels=n_features, kernel_size=(3, 3), padding=1))
        stacker.stack(nn.ReLU())
        self.model, self.output_dim = stacker.get(self.device)

    def encode(self, option: Option[OptionData]) -> torch.Tensor:
        """
        :param option: Option[Point], with depth
        :return: one-hot as image, with depth added to all pixels
            torch.Tensor[dev, f32] : [1, im_x, im_y,]
        """
        result: torch.Tensor = torch.zeros(self.image_shape, dtype=torch.float32, device=self.device)
        # result : Tensor[dev, f32] : [im_x, im_y]
        if option is not None:
            result[tuple(option.value)] = 1.
            result += option.depth
        result = result.unsqueeze(0)
        # [1, x, y]
        return result

    def forward(self, options: List[Option[OptionData]]) -> torch.Tensor:
        """
        list of options to be input, all processed in the same manner
        :param options: options to be encoded (order & length must be fixed)
        :return: encoded: Tensor[dev, f32] : [im_x, im_y, n_features]
        """
        assert len(options) == self.num_inputs
        encodings = [self.encode(option) for option in options]
        x = torch.cat(encodings, dim=0)
        # 【channels, x, y]
        x = x.unsqueeze(0) #add batchdim
        return self.model(x)

def StateEncoder(
        image_shape: Tuple[int, int, int, int],
        n_features: int,
        n_layers: int,
        device: torch.device) -> Tuple[nn.Sequential, List[int]]:
    """

    :param image_shape: (batch, channels, imx, imy)
    :param n_features: number of channels at each layer (besides input)
    :param n_layers: number of layers, 1 gives linear convolution
    :param device: device for model to be placed on
    :return: Model [dev, f32] : image_shape => [dev, f32] : [imx, imy, n_features]
    """
    assert n_layers > 0
    stacker = Stacker(image_shape)
    shape: List[int] = image_shape
    for idx in range(n_layers):
        stacker.stack(nn.Conv2d(shape[1], n_features, kernel_size=(3, 3), padding=1))

        if idx != n_layers - 1:
            shape = stacker.stack(nn.ReLU())

    return stacker.get(device)

def SharedModel(
        input_shape: Tuple[int, int, int, int],
        n_feature: int,
        n_layers: int,
        device: torch.device):
    """

    :param input_shape: [batch, channels, x, y]
    :param n_feature: num features at each layer
    :param n_layers: num layers
    :param device: torch.device
    :return: model: nn.Sequential, shape: List[int]
    """
    stacker = Stacker(input_shape)
    shape: List[int] = list(input_shape)
    shape = stacker.stack(nn.Conv2d(shape[1], n_feature, kernel_size=(3,3)))
    shape = stacker.stack(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    shape = stacker.stack(nn.ReLU())
    shape = stacker.stack(nn.Flatten())
    for idx in range(n_layers):
        features: int = n_feature if idx != n_layers - 1 else 1
        shape = stacker.stack(nn.Linear(shape[1], features))
        if idx != n_layers - 1:
            shape = stacker.stack(nn.ReLU())
    return stacker.get(device)

class ValueModel(nn.Module):
    def __init__(self,
            n_options_input: int,
            image_shape: List[int], # [batch, channels, x, y]
            n_feature: int,
            n_layers: int,
            device: torch.device):
        super().__init__()
        self.option_encoder: nn.Sequential = EncoderModel(image_shape[2:], n_feature * 2, n_options_input, device)
        self.state_encoder, state_embed_size = StateEncoder(image_shape, n_feature, n_layers, device)
        self.value_net,  self.output_shape = SharedModel(state_embed_size, n_feature, n_layers, device)
        self.n_features: int = n_feature
        self.device: torch.device = device

    def forward(self, state: State, options: List[Option]) -> torch.Tensor:
        option_embedding: torch.Tensor = self.option_encoder(options)
        # Tensor : [batch, imx, imy, n_features * 2]
        state: torch.Tensor = torch.from_numpy(state)  # [x, y, channels]
        state = state.to(self.device).float()  # torch.Tensor[dev, f32] : [x, y, channels]
        state = state.permute((2, 0, 1)).unsqueeze(0)  # [batch=1, channels, x, y]
        state_embedding: torch.Tensor = self.state_encoder(state)
        # Tensor: [batch, imx, imy, n_features]
        combined: torch.Tensor = (option_embedding[:, :self.n_features, :, :] * state_embedding)
        combined += option_embedding[:, self.n_features:, :, :]
        # Tensor: [batch, imx, imy, n_features]
        return self.value_net(combined)

class QModel(ValueModel, IQModel[State, Reward, OptionData]):
    def __init__(self,
                 image_shape: List[int],
                 n_feature: int,
                 n_layers: int,
                 device: torch.device):
        super().__init__(3, image_shape, n_feature, n_layers, device)

    def forward(self,
            state: State,
            prev_option: Option[OptionData],
            suboption: Option[OptionData],
            option: Option[OptionData]) -> torch.Tensor:
        return super().forward(state, [prev_option, suboption, option])

class VModel(ValueModel, IVModel[State, Reward, OptionData]):
    def __init__(self,
                 image_shape: List[int],
                 n_feature: int,
                 n_layers: int,
                 device: torch.device):
        super().__init__(2, image_shape, n_feature, n_layers, device)

    def forward(self,
                state: State,
                prev_option: Option[OptionData],
                option: Option[OptionData]) -> torch.Tensor:
        return super().forward(state, [prev_option,  option])


