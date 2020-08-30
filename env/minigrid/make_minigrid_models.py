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
        self.num_inputs: int = num_inputs
        self.image_shape: Tuple[int, int] = image_shape
        self.device: torch.device = device
        stacker = Stacker(input_shape=image_shape)
        stacker.stack(nn.Conv2d(1, out_channels=n_features, kernel_size=(3, 3), padding=1))
        stacker.stack(nn.ReLU())
        self.model, self.output_dim = stacker.get()

    def encode(self, option: Option[OptionData]) -> torch.Tensor:
        """
        :param option: Option[Point], with depth
        :return: one-hot as image, with depth added to all pixels
            torch.Tensor[dev, f32] : [im_x, im_y, 1]
        """
        result: torch.zeros(self.image_shape + (1,), dtype=torch.float32, device=self.device)
        # result : Tensor[dev, f32] : [im_x, im_y, 1]
        result[option.value] = 1.
        result += option.depth
        return result

    def forward(self, options: List[Option[OptionData]]) -> torch.Tensor:
        """
        list of options to be input, all processed in the same manner
        :param options: options to be encoded (order & length must be fixed)
        :return: encoded: Tensor[dev, f32] : [im_x, im_y, n_features]
        """
        assert len(options) == self.num_inputs
        encodings = [self.encode(option) for option in options]
        x = torch.cat(encodings, dim=2)
        return self.model(x)

def StateEncoder(
        image_shape: Tuple[int, int, int],
        n_features: int,
        n_layers: int,
        device: torch.device) -> nn.Sequential:
    """

    :param image_shape: (imx, imy, n_channels)
    :param n_features: number of channels at each layer (besides input)
    :param n_layers: number of layers, 1 gives linear convolution
    :param device: device for model to be placed on
    :return: Model [dev, f32] : image_shape => [dev, f32] : [imx, imy, n_features]
    """
    assert n_layers > 0
    model: nn.Sequential = nn.Sequential()
    in_dims = [image_shape[2]] + [n_features for _ in range(n_layers -1)]
    out_dims = [n_features for _ in range(n_layers)]

    for idx, (indim, outdim) in enumerate(zip(in_dims, out_dims)):
        model.append(nn.Conv2d(indim, outdim, kernel_size=(3, 3), padding=1))

        if idx != n_layers - 1:
            # if not last layer
            model.append(nn.ReLU())

    return model.to(device)

def ValueModel(
        input_shape: Tuple[int, int, int],
        n_feature: int,
        n_layers: int,
        device: torch.device):
    stacker = Stacker(input_shape)
    shape: List[int] = list(input_shape)
    for _ in range(n_layers):
        shape = stacker.stack(nn.Conv2d(shape[1], n_feature, kernel_size=(3,3)))
        shape = stacker.stack(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        shape = stacker.stack(nn.ReLU())
    shape = stacker.stack(nn.Flatten())
    stacker.stack(nn.Linear(shape[1], 1))
    return stacker.get(device)

class ValueModel(nn.Module):
    def __init__(self,
            n_options_input: int,
            image_shape: List[int],
            n_feature: int,
            n_layers: int,
            device: torch.device):
        self.option_encoder: nn.Sequential = EncoderModel(image_shape[:2], n_feature * 2, n_options_input, device)
        self.state_encoder: nn.Sequential = StateEncoder(image_shape, n_feature, n_layers, device)
        state_embed_size: List[int] = Stacker.get_output_shape(image_shape, self.state_encoder)
        self.value_net: nn.Sequential = ValueModel(state_embed_size, n_feature, n_layers, device)
        self.n_features: int = n_feature

    def forward(self, state: State, options: List[Option]) -> torch.Tensor:
        option_embedding: torch.Tensor = self.option_encoder(options)
        # Tensor : [batch, imx, imy, n_features * 2]
        state_embedding: torch.Tensor = self.state_encoder(state)
        # Tensor: [batch, imx, imy, n_features]
        combined: torch.Tensor = (option_embedding[:, :, :, :self.n_features] * state_embedding)
        combined += option_embedding[:, :, :, self.n_features:]
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


