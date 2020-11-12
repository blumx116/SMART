from datetime import datetime
import os

import gym
import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import SMARTAgent, training
from agent.memory import CompleteMemory
from agent.planning_terminator import DepthPlanningTerminator
from agent.policy_terminator import StrictGoalTerminator
from env.minigrid.wrappers import OnehotWrapper
from env.minigrid import MinigridBacktrackingAgent, SimpleMinigridGenerator, VModel, QModel, Evaluator

settings = {
    'random': 2,
    'device': torch.device("cuda:0"),
    'N_EPISODES': 5,
    'TEST_FREQ': 1,
    'VIZ_FREQ': 1,
    'max_depth': 1,
    'environment_name': 'MiniGrid-SimpleCrossingS9N1-v0'
}

runtime = datetime.now().strftime("%Y-%m-%d @ %H-%M-%S")

writer = SummaryWriter(os.path.join("..", "runs", runtime))


env = gym.make(settings['environment_name'])
# env = gym.make("MiniGrid-SimpleCrossingS9N2-v0")
env.seed(settings['random'])
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
env = OnehotWrapper(env)

assert isinstance(env.observation_space, gym.spaces.Box)

low_level_agent = MinigridBacktrackingAgent()
shape = env.observation_space.shape
shape = (-1, shape[-1], shape[0], shape[1])
v_model = VModel(shape, 32, 2, device=settings['device'])
q_model = QModel(shape, 32, 2, device=settings['device'])
planning_terminator = DepthPlanningTerminator(max_depth=settings['max_depth'])
evaluator = Evaluator(v_model, q_model, planning_terminator, settings, get_beta=lambda step: 3, gamma=0.99)
generator = SimpleMinigridGenerator()
memory = CompleteMemory(max_length=100000)
def goal_met(s, o):
    agent_loc: np.ndarray = s[:, :, 8] # imx, imy, onehot
    agent_loc = np.unravel_index(np.argmax(agent_loc), agent_loc.shape)
    return np.all(agent_loc == o.value)
policy_terminator = StrictGoalTerminator(goal_met)
agent = SMARTAgent(
    evaluator,
    generator,
    planning_terminator,
    policy_terminator=policy_terminator,
    low_level=low_level_agent,
    memory=memory,
    settings=settings)



testfn = training.make_simple_minigrid_test(env, writer, range(5))
vizfn = training.make_visualize(env, writer, range(5))

training.train(agent, env, settings, testfn=testfn, vizfn=vizfn)
training.summarize(agent, env, settings, list(range(10)), writer)
