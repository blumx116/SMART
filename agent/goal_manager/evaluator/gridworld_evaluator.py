import math
from typing import List, Tuple, Union 

import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 

from agent.memory import IMemory
from agent.memory.trees import Node
from env.mazeworld import MazeWorld
from misc.typevars import State, Goal, Trajectory, TrainSample
from misc.utils import array_equal

# DEBUG
amax = lambda s: np.unravel_index(np.argmax(s[:,:,-1]), s[:,:,-1].shape)

class GridworldEvaluator:
    def __init__(self, xdim:int, ydim:int, device: torch.device, gamma: float):
        self.device: torch.device = device 
        self.context: torch.Tensor = None 
        self.loss: nn.modules.loss = nn.MSELoss()
        assert 0 < gamma <= 1
        self.gamma: float = gamma 

        conv_channels1 = 20
        conv_size1 = 3
        maxpool_size1 = 2
        conv_channels2 = 15
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
            nn.LayerNorm(hidden_input),
            nn.Linear(hidden_input, 50),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(50),
            nn.Linear(50, 1)).to(self.device)

        self.target: nn.Sequential = nn.Sequential(
            nn.Conv2d(4, conv_channels1, conv_size1),
            nn.MaxPool2d(maxpool_size1),
            nn.Conv2d(conv_channels1, conv_channels2, conv_size2),
            nn.Flatten(),
            nn.LayerNorm(hidden_input),
            nn.Linear(hidden_input, 50),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(50),
            nn.Linear(50, 1)).to(self.device)

        self.optimizer = optim.SGD(self.inner.parameters(), lr=1e-3)


    def estimate_path_reward(self, state: Union[State, Goal],
        goal: Union[Goal, Node[Goal]], network: str = "inner") -> torch.Tensor:
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

        input = np.concatenate((goal, state), axis=-1)
        # np.ndarray[int]: [y_dim, x_dim, 2]
        input = torch.from_numpy(input).float().to(self.device)
        # torch.Tensor[float, device] : [y_dim, x_dim, 2]
        input = torch.cat((self.context, input), dim=2)
        # torch.Tensor[float, device] : [y_dim, x_dim, 4]
        input = input.permute((2, 0, 1)).unsqueeze(0).contiguous()
        # receive CUDNN_STATUS_NOT_SUPPORTED error if remove contiguous
        # torch.Tensor[float, device] : [1, 4, y_dim, x_dim]

        network: nn.Module = self.get_network(network)
        return network(input).squeeze(0)

    def get_network(self, network: str) -> nn.Module:
        network = network.lower()
        assert network in ['inner', 'target']
        lookup = {
            "inner" : self.inner,
            "target": self.target
        }
        return lookup[network]

    def print_coord(self, arr, dim):
        arr = arr[:,:,dim]
        print(np.unravel_index(arr.argmax(), arr.shape))

    def estimate_subpath_reward(self, state: State, subgoal: Goal, 
        goal: Goal, network: str = "inner") -> torch.Tensor:
        return self.estimate_path_reward(state, subgoal, network) + \
            self.estimate_path_reward(subgoal, goal, network)

    def score_subgoals(self, subgoals: List[Goal], state: State,
        goal: Goal, network: str = "inner") -> np.ndarray:

        result: Iterable[torch.Tensor] = map(
            lambda subgoal: self.estimate_subpath_reward(state, subgoal, goal, network), 
            subgoals)
        #Iterable of scalars on device
        result: Iterable[float] = map(lambda tens: tens.item(), result)
        return np.asarray(list(result), dtype=float)

    def selection_probabilities(self, subgoals: List[Goal], scores: np.ndarray, 
        state: State, goal: Goal) -> np.ndarray:
        weights: np.ndarray = np.exp(scores)
        return weights / weights.sum()

    def choose_subgoal(self, 
        possible_subgoals: List[Goal], 
        state: State, 
        goal_node: Node[Goal], 
        network: str="inner") -> Tuple[Goal, np.ndarray]:
        # returns chosen goal and scores
        scores: np.ndarray = self.score_subgoals(possible_subgoals, state, goal_node.value, network)
        probabilities: np.ndarray = self.selection_probabilities(possible_subgoals, scores, state, goal_node.value)
        chosen_idx: int = np.random.choice(range(len(probabilities)), p=probabilities)
        return possible_subgoals[chosen_idx],  scores

    def reset(self, env: MazeWorld, goal: Goal) -> None:
        self.context = torch.from_numpy(env._grid).float().to(self.device)

    def _trajectory_reward(self, trajectory: Trajectory) -> torch.Tensor:
        result: Reward = 0
        discount: float = 1
        for transition in trajectory:
            result += discount * transition.reward 
            discount *= self.gamma
        return torch.tensor(result, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _piecewise_trajectory_estimate(self, sample: TrainSample, network: str = 'inner') -> torch.Tensor:
        return self.estimate_path_reward(sample.initial_state, sample.subgoal.value, network) + \
            self.estimate_path_reward(sample.subgoal.value, sample.goal.value, network)  


    def optimize(self, samples: List[TrainSample]) -> None:
        if samples is None or len(samples) == 0:
            return
        truths: List[torch.Tensor] = [ ]
        preds: List[torch.Tensor] = [ ]
        for sample in samples:
            preds.append(
                self.estimate_path_reward(
                    state=sample.initial_state, 
                    goal=sample.goal,
                    network="inner"))
            truths.append(
                self._trajectory_reward(
                    sample.subgoal_trajectory + sample.goal_trajectory))
            """
            if array_equal(sample.subgoal, sample.goal):
                truths.append(self._trajectory_reward(sample.subgoal_trajectory))
            else:
                with torch.no_grad():
                    truths.append(
                        self._piecewise_trajectory_estimate(
                            sample=sample, 
                            network='target'))
            """
        print("==================={step}================")
        for sample in samples:
            print(f"sample: {amax(sample.initial_state)}=>{amax(sample.goal.value)} (len: {len(sample.subgoal_trajectory)+len(sample.goal_trajectory)})")
        truths: torch.Tensor = torch.cat(truths)
        preds: torch.Tensor = torch.cat(preds)
        print(f"truths: {truths}")
        print(f"preds: {preds}")
        # torch.Tensor[float, device] : [len(samples), ]
        loss = self.loss(preds, truths)
        print(f"loss: {loss}")
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        #double check
        preds = [] 
        for sample in samples:
            preds.append(
                self.estimate_path_reward(
                    state=sample.initial_state, 
                    goal=sample.goal,
                    network="inner"))
        preds: torch.Tensor = torch.cat(preds)
        loss = self.loss(preds, truths)
        self.optimizer.zero_grad()
        print(f"new preds: {preds}")
        print(f"new loss: {loss}")
        print("==============={end step}=====================")
