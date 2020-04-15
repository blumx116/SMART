from env.mazeworld.maze_world_generator import MazeWorldGenerator
from env.mazeworld.maze_world_cache import MazeWorldCache
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle as pkl 
import dill

generator = MazeWorldGenerator(17, 17, 10,  100, 15)
cache = MazeWorldCache(generator)
for i in range(50):
    
    gridworld: np.ndarray = cache._get_cached_board(i)

    plt.imshow(gridworld[:,:,0] + (2 * gridworld[:,:,1]))
    plt.show()
    