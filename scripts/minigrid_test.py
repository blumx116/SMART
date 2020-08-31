import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
import torch

from agent import SMARTAgent
from agent.evaluator import AEvaluator
from agent.memory import CompleteMemory
from agent.planning_terminator import DepthPlanningTerminator
from agent.policy_terminator import StrictGoalTerminator
from env.minigrid.wrappers import OnehotWrapper, find, onehot2directedpoint
from env.minigrid import MinigridBacktrackingAgent, SimpleMinigridGenerator, VModel, QModel, Evaluator
from misc.typevars import Option

states = []
initials = []

settings = {
    'random' : 2,
    'device' : torch.device("cuda:0")
}

N_EPISODES = 10
env = gym.make('MiniGrid-LavaGapS7-v0')
# env = gym.make("MiniGrid-SimpleCrossingS9N2-v0")
env.seed(settings['random'])
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
env = OnehotWrapper(env)

env.render()

assert isinstance(env.observation_space, gym.spaces.Box)

low_level_agent = MinigridBacktrackingAgent()
shape = env.observation_space.shape
shape = (-1, shape[-1], shape[0], shape[1])
v_model = VModel(shape, 32, 2, device=settings['device'])
q_model = QModel(shape, 32, 2, device=settings['device'])
planning_terminator = DepthPlanningTerminator(max_depth=3)
evaluator = Evaluator(v_model, q_model, planning_terminator, settings, get_beta=lambda step: 3, gamma=0.99)
generator = SimpleMinigridGenerator()
memory = CompleteMemory(max_length=100000)
def goal_met(s, o):
    agent_loc: np.ndarray = s[:,:, 8] # imx, imy, onehot
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

def visualize(rgb_array, options):
    #tiles of size 32 x 32
    for option in options:
        tile_ur: np.ndarray = option.value.astype(np.int32) * 32
        tile_ur = tile_ur[::-1]
        for y in range(4, 28):
            for x in range(4, 28):
                new_x, new_y = tile_ur + np.asarray([x, y])

                colors = [[255, 0, 0], [255, 102, 102], [255, 128, 0], [255, 178, 102], [255, 255, 0], [255, 255, 102]]
                rgb_array[new_x, new_y, :] = np.asarray(colors[option.depth])
    return rgb_array

def get_option_tree(agent):
    result = []
    option_node = agent.current_option_node
    prev_option = None
    while option_node is not None:
        if prev_option is None or option_node.left == prev_option:
            result.append(option_node.value)
        prev_option = option_node
        option_node = option_node.parent
    return result

images = []
for _ in range(N_EPISODES):
    state = env.reset()
    goal_point = find(state, 'Goal')
    option = Option(goal_point, depth=0)
    agent.reset(env, option, random_seed=3)

    images.append([plt.imshow(env.render('rgb_array'), animated=True)])
    done = False
    while not done:
        action = agent.act(state, option)
        state, reward, done, _ = env.step(action)
        options = get_option_tree(agent)
        print(f"@{onehot2directedpoint(state)} : {reward} => {options}")
        rendered = visualize(env.render('rgb_array'), options)
        images.append([plt.imshow(rendered, animated=True)])

fig = plt.figure()
ani = animation.ArtistAnimation(fig, images, interval=100,blit=True, repeat=True)
plt.show()

import time
while True:
    time.sleep(10000)