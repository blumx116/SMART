# from env.minigrid.wrappers import OnehotWrapper, Onehot2PointWrapper

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym_minigrid import register
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from env.minigrid.wrappers import OnehotWrapper, find, onehot2directedpoint
from env.minigrid import MinigridBacktrackingAgent
from misc.typevars import Option
"""
env = gym.make("MiniGrid-LavaGapS5-v0")
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
env = OnehotWrapper(env)
env = Onehot2PointWrapper(env)
state = env.reset()
state2 = env.step(0)
state3 = env.step(1)

env.render('human')
state = env.reset()
print()
"""

states = []
initials = []
env = gym.make("MiniGrid-SimpleCrossingS9N2-v0")
env.seed(10)
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
env = OnehotWrapper(env)

env.render()

agent = MinigridBacktrackingAgent()

state = env.reset()
goal_point = find(state, 'Goal')
option = Option(goal_point, depth=0)
images = []
images.append([plt.imshow(env.render('rgb_array'), animated=True)])
done = False
while not done:
    action = agent.act(state, option)
    state, reward, done, _ = env.step(action)
    print(f"@{onehot2directedpoint(state)} : {reward}")
    images.append([plt.imshow(env.render('rgb_array'), animated=True)])


fig = plt.figure()
ani = animation.ArtistAnimation(fig, images, interval=100,blit=True, repeat=True)
plt.show()

print()