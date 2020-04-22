import matplotlib.pyplot as plt

from env.mazeworld import MazeWorldCache, MazeWorldGenerator, MazeWorld
from agent.simple.greedy_maze_agent import GreedyMazeAgent

generator = MazeWorldGenerator(10, 10, 2,  100, 10)
cache = MazeWorldCache(generator)
env = MazeWorld(cache._get_cached_board(0))

state, goal = env.reset(3)
agent = GreedyMazeAgent(env)
done = False 
while not done:
    print('step')
    action = agent.act(state, goal)
    state, reward, done, info = env.step(action)
    print(env._location)