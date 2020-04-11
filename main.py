from mazeworld import MazeWorld
from utils import array_unique
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle as pkl 
import dill

world = MazeWorld(17, 17, 10,  100, 15)

for i in range(67, 150):
    gridworld: np.ndarray = world.reset()
    # np.save(f"gridworld_{i}.csv", gridworld)

    plt.imshow(gridworld[:,:,0] + (2 * gridworld[:,:,1]))
    plt.show()