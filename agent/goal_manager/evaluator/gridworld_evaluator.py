import math
from typing import List, Tuple

import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 

from agent.goal_manager.memory_manager import IMemoryManager
from agent.memory.trees import Node
from misc.typevars import State, Goal 

class GridworldEvaluator:
    def __init__(self, xdim:int, ydim:int, device):
        self.device = device 
        self.context = None 
        self.loss = nn.MSELoss()

        conv_channels1 = 7
        conv_size1 = 3
        maxpool_size1 = 2
        conv_channels2 = 3
        conv_size2 = 3


        #equation from https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        conv1_x = xdim - (conv_size1 - 1)
        conv1_y = ydim - (conv_size1 - 1)
        maxpool1_x = math.floor(conv1_x / maxpool_size1)
        maxpool1_y = math.floor(conv1_y / maxpool_size1)
        conv2_x = maxpool1_x - (conv_size2 - 1)
        conv2_y = maxpool1_y - (conv_size2 - 1)
        hidden_input = conv2_x * conv2_y * conv_channels2


        self.inner: nn.Sequential = nn.Sequential(
            nn.Conv2d(4, conv_channels1, conv_size1),
            nn.MaxPool2d(maxpool_size1),
            nn.Conv2d(conv_channels1, conv_channels2, conv_size2),
            nn.Flatten(),
            nn.Linear(hidden_input, 50),
            nn.LeakyReLU(0.1),
            nn.Linear(50, 1)).to(self.device)

        self.optimizer = optim.SGD(self.inner.params(), lr=0.01)


    def estimate_path_reward(self, state: Union[State, Goal],
        goal: Union[Goal, Node[Goal]]) -> torch.Tensor:
        """
            state: [y_dim, x_dim, 3] or [y_dim, x_dim, 1]
            goal: [y_dim, x_dim, 1]
            result: torch.Tensor[float, device] : [1,]
        """
        if state.shape[2] > 1:
            state = state[:,:,-1] #[y_dim, x_dim]
            state = state[:,:,np.newaxis] #[y_dim, x_dim, 1]
        if isinstance(goal, Node):
            goal = goal.value # [y_dim, x_dim, 1]

        input = np.stack((goal, state), dim=-1)
        # np.ndarray[int]: [y_dim, x_dim, 2]
        input = torch.from_numpy(input).float().to(self.device)
        # torch.Tensor[float, device] : [y_dim, x_dim, 2]
        input = torch.cat((self.context, input), dim=2)
        # torch.Tensor[float, device] : [y_dim, x_dim, 4]
        return self.inner(input).squeeze(0)

    def estimate_subpath_reward(self, state: State, subgoal: Goal, 
        goal: Goal) -> torch.Tensor:
        return self.estimate_path_reward(state, subgoal) + \
            self.estimate_path_reward(subgoal, goal)

    def score_subgoals(self, subgoals: List[Goal], state: State,
        goal: Goal) -> np.ndarray:

        result: Iterable[torch.Tensor] = map(lambda subgoal: self.estimate_subpath_reward(
            state, subgoal, goal), subgoals)
        #Iterable of scalars on device
        result: Iterable[float] = map(lambda tens: tens.item(), result)
        return np.asarray(list(result), dtype=float)

    def selection_probabilities(self, subgoals: List[Goal], scores: np.ndarray, 
        state: State, goal: Goal) -> np.ndarray:
        weights: np.ndarray = np.exp(scores)
        return weights / weights.sum()

    def choose_subgoal(self, possible_subgoals: List[Goal], 
        state: State, goal_node: Node[Goal]) -> Tuple[Goal, np.ndarray]:
        # returns chosen goal and scores
        scores: np.ndarray = self.score_subgoals(possible_subgoals, state, goal_node.value)
        probabilities: np.ndarray = self.selection_probabilities(possible_subgoals, scores, state, goal_node.value)
        return np.random.choice(range(len(probabilities)), p=probabilities), scores

    def reset(self, env: Environment, goal: Goal) -> None:
        self.context = torch.from_numpy(env._grid).float().to(self.device)

    def step(self, memory_manager: IMemoryManager) -> None:
        if len(memory_manager) == 0:
            return
        truths: List[float] = []
        preds: List[torch.Tensor] = [] 
        for i in range(50):
            state, goal, target = memory_manager.generate_sample()
            truths.append(target)
            preds.append(self.estimate_path_reward(state, goal))
        truths: torch.Tensor = torch.FloatTensor(truths).to(self.device)
        # torch.Tensor[float, device] : [50,]
        preds:  torch.Tensor = torch.cat(preds)
        loss = self.loss(preds, truths)
        loss.backward()
        self.optimizer.step()

