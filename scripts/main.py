from env.mazeworld.mazeworld_generator import MazeWorldGenerator
from env.mazeworld.mazeworld_cache import MazeWorldCache
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle as pkl 
import dill

generator = MazeWorldGenerator(10, 10, 2,  100, 10)
cache = MazeWorldCache(generator)
for i in range(100):
    
    gridworld: np.ndarray = cache._get_cached_board(i)

    #plt.imshow(gridworld[:,:,0] + (2 * gridworld[:,:,1]))
    # plt.show()
    