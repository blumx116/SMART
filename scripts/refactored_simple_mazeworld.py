from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import SMARTAgent, IOptionBasedAgent
from agent.evaluator import IEvaluator, IVModel, IQModel
from agent.generator import IGenerator 
from agent.memory import IMemory, CompleteMemory
from agent.planning_terminator import IPlanningTerminator, DepthPlanningTerminator
from agent.policy_terminator import IPolicyTerminator, StrictGoalTerminator
from env.mazeworld import MazeWorld, MazeWorldCache, MazeWorldGenerator
from env.mazeworld.agent import SimpleMazeworldGenerator, SimpleMazeworldEvaluator
from env.mazeworld.agent import MazeworldQModel, MazeworldVModel, BacktrackingMazeAgent
from misc.typevars import State, Action, Reward, Transition, Option


settings = {
    'tensorboard' : SummaryWriter(),
    'device' : torch.device("cuda:0"),
    'random' : 3
}

YDIMS = 10
XDIMS = 10
generator = MazeWorldGenerator(YDIMS, XDIMS, 2,  100, 10)
cache = MazeWorldCache(generator)
global env
env = MazeWorld(cache._get_cached_board(0))

goal_achieved =  lambda state, option: np.array_equal(env._grid_to_point(state[:,:,2]), option)
policy_terminator: IPolicyTerminator = StrictGoalTerminator(goal_achieved)

planning_terminator: IPlanningTerminator = DepthPlanningTerminator(max_depth=1)

v_model: IVModel = MazeworldVModel(XDIMS+2, YDIMS+2, settings)
q_model: IQModel = MazeworldQModel(XDIMS+2, YDIMS+2, settings)
get_beta = lambda step: 0.001 * step
evaluator: IEvaluator = SimpleMazeworldEvaluator(planning_terminator, v_model, q_model, settings, get_beta, gamma=0.99)

generator: IGenerator = SimpleMazeworldGenerator()

low_level: IOptionBasedAgent = BacktrackingMazeAgent(env)

memory: IMemory = CompleteMemory(max_length=100, random_seed=settings['random'])

agent: SMARTAgent = SMARTAgent(
    evaluator,
    generator,
    planning_terminator,
    policy_terminator,
    low_level,
    memory,
    settings
)

step: int = 0
images = [] 
for seed in [0] * 500:
    env = MazeWorld(cache._get_cached_board(seed))

    total_reward: int = 0
    t: int = 0
    done: bool = False 

    state, goal = env.reset(3)
    goal = Option(goal, 0)
    states: List[State] = state
    agent.reset(env, goal)

    while not done:
        print('step') 
        action: Action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        total_reward += reward 
        states.append(Transition(state, action, reward, new_state))
        state = new_state
        agent.optimize(step)
        t += 1
        step += 1


    def render(env: MazeWorld, state: State):
        grid: np.ndarray = env._grid
        # np.ndarray[float64] : [YDIMS, XDIMS, 2]
        image: np.ndarray = grid[:,:,0] + (2 * grid[:,:,1])
        image += (state[:,:,2] * 3)
        image += env._goal_as_grid[:,:,0] * 4
        # np.ndarray[float64] : [YDIMS, XDIMS]
        return plt.imshow(image, animated=True)

    for state in states:
        images.append([render(env, state)])

fig = plt.figure()
ani = animation.ArtistAnimation(fig, images, interval=100,blit=True, repeat=False)
plt.show()
