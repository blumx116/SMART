import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from agent import SMARTAgent
from agent.memory import CompleteMemory
from agent.planning_terminator import DepthPlanningTerminator
from env.minigrid.wrappers import OnehotWrapper, find, onehot2directedpoint
from env.minigrid import MinigridBacktrackingAgent, SimpleMinigridGenerator
from misc.typevars import Option

states = []
initials = []

settings = {
    'random' : 0
}

env = gym.make("MiniGrid-SimpleCrossingS9N2-v0")
env.seed(10)
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
env = OnehotWrapper(env)

env.render()

low_level_agent = MinigridBacktrackingAgent()
evaluator =
generator = SimpleMinigridGenerator()
planning_terminator = DepthPlanningTerminator(max_depth=2)
memory = CompleteMemory(max_length=100000)
high_level_agent = SMARTAgent()

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