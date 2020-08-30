import math
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

from agent.evaluator import IQModel
from env import IEnvironment
from misc.typevars import State, Action, Reward, Option, OptionData

class EncoderModel(nn.Module):
    def __init__(self,
            image_shape: Tuple[int, int],
            n_features: int,
            num_inputs: int,
            device: torch.device):
        self.num_inputs: int = num_inputs
        self.image_shape: Tuple[int, int] = image_shape
        self.device: torch.device = device
        self.model = nn.Sequential(
            nn.Conv2d(1, out_channels=n_features, kernel_size=(3, 3), padding=1),
            nn.ReLU()).to(self.device)

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


class MinigridQModel(IQModel[State, Action, Reward]):
   def __init__(self,
        input_shape: Tuple[int, int, int],
        n_features: int,
        n_layers: int,
        device: torch.device):
       self.option_encoder: nn.Sequential = EncoderModel(input_shape[:2], n_features * 2, num_inputs=3, device=device)
       # [Option, Option, Option] => Tensor[dev, f32] : [imx, imy, 2*n_features]
       self.state_encoder: nn.Sequential = StateEncoder(input_shape, n_features, n_layers, device=device)
       # Tensor[dev, f32]: input_shape => Tensor[dev, f32] : [imx, imy, n_features]
       self.value_function =


