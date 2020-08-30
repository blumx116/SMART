from typing import List, Any, Dict 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch 
import torchvision

from agent.simple import BacktrackingMazeAgent, Grid2PointWrapper
from agent import SMARTAgent, IAgent
from agent.goal_manager import SimpleGoalManager, IGoalManager
from agent.goal_manager.evaluator import GridworldEvaluator, IEvaluator
from agent.goal_manager.generator import SimpleGridworldGenerator, IGenerator
from agent.memory import IMemory, CompleteMemory
from env.mazeworld import MazeWorldCache, MazeWorldGenerator, MazeWorld, State
from misc.utils import array_equal

YDIMS = 10
XDIMS = 10
generator = MazeWorldGenerator(YDIMS, XDIMS, 2,  100, 10)
cache = MazeWorldCache(generator)
global env
env = MazeWorld(cache._get_cached_board(0))
device: torch.device = torch.device("cuda:0")

settings : Dict[str, Any] = {
    "device" : torch.device("cuda:0"),
    "tensorboard" : SummaryWriter()
}

fig = plt.figure() 
images = [] 

low_level_agent: IAgent = BacktrackingMazeAgent(env)
low_level_agent: IAgent = Grid2PointWrapper(low_level_agent)
evaluator: IEvaluator = GridworldEvaluator(XDIMS+2, YDIMS+2, settings, gamma=0.99)
generator: IGenerator = SimpleGridworldGenerator()
fulfils_goal = lambda state, goal: array_equal(state[:,:,-1], goal[:,:,0])
goal_manager: IGoalManager = SimpleGoalManager(evaluator, generator, 1, fulfils_goal)
memory: IMemory = CompleteMemory(100, 3)
agent = SMARTAgent(goal_manager, low_level_agent, memory)

totals = [] 

step: int = 0


for iter, seed in enumerate([0] * 500):
    total_reward: int = 0
    print(f"================={seed}=================")
    env= MazeWorld(cache._get_cached_board(seed))

    state, goal = env.reset(3)
    agent.reset(env, state, goal)
    done = False 
    states: List[State] = [state]

    t: int = 0

    while not done:
        print('step')
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        states.append(state)
        agent.view(state, action, reward)
        print(env._location)

        agent.optimize(step)
        t += 1
        step += 1

    settings['tensorboard'].add_scalar('duration', t, iter)
    settings['tensorboard'].add_scalar('reward', total_reward, iter)
    totals.append(total_reward)

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

ani = animation.ArtistAnimation(fig, images, interval=100,blit=True, repeat=False)
plt.show()
