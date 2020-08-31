import numpy as np

Point = np.ndarray # np.ndarray[int8] : [2,] => (x, y) -> y, x???
OneHotImg = np.ndarray # np.ndarray[int8] : [xdim, ydim, 15]
RawState = np.ndarray  # np.ndarray[int8] : [xdim, ydim, 3]
DirectedPoint = np.ndarray # np.ndarray[int8] : [4, ] => (x, y, dx, dy))
Action = int # range(7)
Reward = int
State = OneHotImg
Goal = Point
