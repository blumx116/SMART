from abc import abstractmethod
from typing import Dict, Any, Callable, List, Union, Tuple, Optional

import numpy as np
from numpy.random import RandomState 
import torch 

from agent.evaluator import IEvaluator, IVModel, IQModel
from env import IEnvironment
from misc.typevars import State, Action, Reward, Option, OptionData
from misc.typevars import TrainSample, Trajectory
from misc.utils import array_random_choice, optional_random

class AEvaluator(IEvaluator[State, Action, Reward, OptionData]):
    def __init__(self, 
        v_model: IVModel[State, Reward, OptionData],
        q_model: IQModel[State, Reward, OptionData],
        settings: Dict[str, Any], 
        get_beta: Callable[[int], float],
        gamma: float):

        self.v_model: IVModel[State, Reward, OptionData] = v_model
        self.q_model: IQModel[State, Reward, OptionData] = q_model
        self.get_beta: Callable[[int], float] = get_beta
        self.step: int = 0
        self.gamma: float = gamma
        self.random: RandomState = optional_random(settings['random'])

    def optimize(self, 
            samples: List[TrainSample[State, Action, Reward, OptionData]], 
            step: int = None) -> None:
        """
            Optimization step. Trains the internal machine learning models
            based on the training samples provided. Always uses all samples.
            Parameters
            ----------
            samples: List[TrainSample]
                the samples to be used to train the models
            step: int
                used only for tensorboard logging
        """
        self._train_q_model_(samples, step)
        self._train_v_model_(samples, step)
    
    def reset(self, 
            env: IEnvironment[State, Action, Reward],
            random_seed: Union[int, RandomState] = None) -> None:
        """
            Prepares the agent to begin functioning in the new environment.
            Should be called each time there is a new episode or a new env.
            random_seed may be passed in as well to set the random seed
            Parameters
            ----------
            env: Environment
                the new environment (may be the same as the old)
            random_seed: Option[Union[int, RandomState]] = None
                the new random seed to be used. If None, no changes
                are made
        """
        if random_seed is not None:
            self.random = optional_random(random_seed)
        self.q_model.reset(env, random_seed)
        self.v_model.reset(env, random_seed)

    def select(self, 
            state: State, 
            possibilities: List[Option[OptionData]],
            prev_option: Optional[Option[OptionData]],
            parent_option: Option[OptionData]) -> Option[OptionData]:
        """
            Conditioned on the current state and option, chooses the next 
            suboption to pursue as a suboption to 'option' from the list of 
            possibilities
            Parameters
            ----------
            state: State
                the state that the agent is at when the chosen option will
                begin to be pursued
            possibilities: List[Option[OptionData]]
                the possible suboptions to pursue. Length must be at least 1
            prev_option: Optional[Option[OptionData]]
                the last option that was pursue. Will be None at start of
                new episode
            parent_option: Option[OptionData]
                the parent option that the chosen option will be a direct
                child of
            Returns
            -------
            chosen: Option
                the option that was chosen to pursue next
        """
        probabilities: np.ndarray = self._selection_probabilities_(
            state, possibilities, prev_option, parent_option,
            normalize=False)
        # np.ndarray[float]: [len(possibilities), ]
        return array_random_choice(possibilities, probabilities, self.random)

    def _score_possibilities_(self,
                              state: State,
                              possibilities: List[Option[OptionData]],
                              prev_option: Optional[Option[OptionData]],
                              parent_option: Option[OptionData]) -> List[float]:
        return list(map(
            lambda possibility: self.q_model.forward(state, prev_option, possibility, parent_option),
            possibilities))

    def _selection_probabilities_(self,
                                  state: State,
                                  possibilities: List[Option[OptionData]],
                                  prev_option: Optional[Option[OptionData]],
                                  parent_option: Option[OptionData],
                                  normalize: bool = False) -> np.ndarray:
        scores: List[float] = self._score_possibilities_(
            state, possibilities, prev_option, parent_option)
        scores: np.ndarray = np.asarray(scores, dtype=float)
        # np.ndarray[float]: [len(possibilities), ]
        probabilities: np.ndarray = np.exp(self.get_beta(self.step) * scores)
        if normalize:
            assert np.all(probabilities > 0)
            probabilities /= probabilities.sum()
        return probabilities



    def _get_v_as_target_(self,
            state: State,
            prev_option: Optional[Option[OptionData]],
            option: Option[OptionData], 
            trajectory: Trajectory[State, Action, Reward]) -> torch.Tensor:
        """
            Uses the VModel as the target for the state, option 
            pair, sometimes using the raw data from the trajectory
            Parameters
            ----------
            state: State
                the initial state of the the trajectory
            prev_option: Optional[Option]
                the last option that had been pursued when 'option' was chosen
            option: Option
                the parent option that was pursued
            trajectory: Trajectory
                the trajectory that was pursued from state->suboption->option
            Returns
            -------
            target: torch.Tensor[float, v_model.device] : [1,]
                the estimated trajectory reward
        """
        if self._should_use_raw_(state, prev_option, option):
            result: torch.Tensor = self._trajectory_reward_(trajectory)
        else:
            result: torch.Tensor = self.v_model.forward(state, prev_option, option)
        return result.to(self.q_model.device)

    def _get_q_as_target_(self,
            state: State,
            prev_option: Optional[Option[OptionData]],
            suboption: Option[OptionData], 
            option: Option[OptionData], 
            trajectory1: Trajectory[State, Action, Reward],
            trajectory2: Trajectory[State, Action, Reward]) -> torch.Tensor:
        """
            Uses the QModel as the target for the state, suboption, option 
            triple, sometimes using the raw data from the trajectory
            Parameters
            ----------
            state: State
                the initial state of the the trajectory
            prev_option: Optional[Option]
                the last option that was pursue before 'option' was started
            suboption: Option
                the suboption that was pursued right away
            option: Option
                the parent option that was pursued
            trajectory1: Trajectory
                the trajectory that was pursued from state->suboption
            trajectory2: Trajectory
                the trajectory that was pursued from suboption->option
            Returns
            -------
            target: torch.Tensor[float, v_model.device] : [1,]
                the estimated trajectory reward
        """
        if self._should_use_raw_(state, option):
            result: torch.Tensor = self._trajectory_reward_(trajectory1 + trajectory2)
        else:
            result: torch.Tensor = self.q_model.forward(
                state, prev_option, suboption, option)
        return result.to(self.v_model.device)

    @abstractmethod
    def _should_use_raw_(self, 
            state: State,
            prev_option: Optional[Option[OptionData]],
            option: Option[OptionData]) -> bool:
        """
            Returns whether or not the model should bootstrap the estimate
            for this sample or use direct training data. If True, uses raw
            trajectory rewards. If False, uses v_model or q_model as to create
            the target.
            Parameters
            ----------
            state: State
                the initial state of the trajectory
            option: Option
                the option that is being pursued and will be input to the model
            Returns
            -------
            raw?: bool
                whether or not to use raw data for training
        """
        pass

    def _train_q_model_(self, 
            samples: List[TrainSample[State, Action, Reward, OptionData]],
            step: int = None) -> None:
        """
            Trains the Q-model to learn that the value of a trajectory
            is the value of the sub-trajectories summed
            Parameters
            ----------
            samples: List[TrainSample],
                the samples to be used for gradient descent. All samples used
            step: Optional[int] = None
                only used for tensorboard logging
        """
        inputs: List[Tuple[State, Option[OptionData], Option[OptionData]]] = list(map(
            lambda sample: (sample.initial_state, sample.suboption, sample.option),
            samples))
        targets: List[torch.Tensor] = []
        for sample in samples:
            first_traj_reward: torch.Tensor = self._get_v_as_target_(
                sample.initial_state, 
                sample.suboption,
                sample.suboption_trajectory)
            second_traj_reward: torch.Tensor = self._get_v_as_target_(
                sample.midpoint_state,
                sample.option,
                sample.option_trajectory)
            #both torch.Tensor[float, q_model.device] : [1,]
            second_traj_reward *= np.float_power(
                self.gamma, 
                len(sample.suboption_trajectory))
            #discount the second trajectory
            targets.append(first_traj_reward + second_traj_reward)
        self.q_model.optimize(inputs, targets, step)

    def _train_v_model_(self, 
            samples: List[TrainSample[State, Action, Reward, OptionData]], 
            step: int = None) -> None:
        """
            Trains the V-model to learn that the V-model is the expectation
            of the Q-model over suboptions using historical data to sample
            suboptions.
            Parameters
            ----------
            samples: List[TrainSample],
                the samples to be used for gradient descent. All samples used
            step: Optional[int] = None
                only used for tensorboard logging
        """
        inputs: List[IVModel.TrainingDatum] = list(map(
            lambda sample: (sample.initial_state,
                            sample.prev_option,
                            sample.option),
            samples))
        targets: List[torch.Tensor] = list(map(
            lambda sample: self._get_q_as_target_(
                sample.initial_state,
                sample.prev_option,
                sample.suboption,
                sample.option,
                sample.suboption_trajectory,
                sample.option_trajectory),
            samples))
        self.v_model.optimize(inputs, targets, step)

    def _trajectory_reward_(self,
            traj: Trajectory[State, Action, Reward]) -> torch.Tensor:
        """
        Sums the discounted rewards received along the trajectory
        Note: reward_dims = 1 if reward is a float
        :param traj: The trajectory observed, only rewards relevant
        :return: torch.Tensor[float32, self.device] : [reward_dims]
        """
        rewards: List[Reward] = list(map(lambda trans: trans.reward, traj))
        rewards: np.ndarray[float] = np.asarray(rewards).astype(float)
        # if reward = float, np.ndarray[float] : [len(traj), ]
        # if reward = array, np.ndarray[float] : [len(traj), reward_dims]
        dts: np.ndarray[int] = np.arange(rewards.shape[0])
        # np.ndarray = [0, 1, 2, 3, 4]
        discounts: np.ndarray[float] = self.gamma ** dts
        # np.ndarray[float] : [len(traj)] e.g. [1, 0.9, 0.81, 0.729,...]
        discounted_rewards: np.ndarray[float] = (rewards.T * discounts).T
        # np.ndarray[float] : [len(traj), reward_dims]
        total_rewards: np.ndarray = np.asarray(np.sum(discounted_rewards, axis=0))
        # np.ndarray[float] : [reward_dims], reward_dims=1 if float
        return torch.from_numpy(total_rewards).type(torch.float32).to(self.device)


