import os
from typing import List, Iterable, TypeVar, Union, Tuple

import numpy as np

T = TypeVar("T")

def flatmap(data: Iterable[Iterable[T]]) -> List[T]:
    return [el for l in data for el in l]

def unpack_tuple(tup: Tuple[T]) -> Union[T, Tuple[T]]:
    if not isinstance(tup, tuple):
        return tup #only unpack tuples
    elif len(tup) > 1:
        return tup
    elif len(tup) == 1:
        return tup[0]
    else: #len == 0
        return None 


def array_unique(data: Iterable[np.ndarray], return_inverse=False, 
        return_counts = False) -> List[np.ndarray]:
    """
        used for essentially calling np.unique on a list of np.ndarrays
        data: Iterable[np.ndarray] : n_data => [ndarray_dims]
        returns : List[np.ndarray] : n_unique => [ndarray_dims]
            if return_inverse : np.ndarray[int] : [n_data] # see np.unique docs
            if return_counts  : np.ndarray[int] : [n_unique] # ditto
    """
    data : np.ndarray = np.vstack(data) # [n_data, ndarray_dims]
    filtered, inv_ret, count_ret = np.unique(data, axis=0, return_inverse=True, return_counts=True) 
    # np.ndarray : [n_unique, ndarray_dims], np.ndarray : [n_unique,], np.ndarray : [n_data,]
    result = [filtered[i] for i in range(filtered.shape[0])]
    # result: List[np.ndarray] : n_unique => [ndarray_dims]
    result = (result, )
    if return_inverse:
        result = result + (inv_ret, )
    if return_counts:
        result = result + (count_ret, )
    return unpack_tuple(result)

def array_contains(el: np.ndarray, list: Iterable[np.ndarray]) -> bool:
    """
        returns whether or not el is in list
    """
    for element in list:
        if np.array_equal(el, element):
            return True
    return False

def array_random_choice(elems: Iterable[np.ndarray], random: np.random.RandomState = None) -> np.ndarray:
    """
        elems : elements to choose from among
        random : random state if desired, uses default random if not
        returns a random array from the list of arrays, used because
        np.random.choice only accepts 1D arrays
    """
    elems = list(elems)
    if random is None:
        random = np.random
    idx = random.choice(len(elems))
    return elems[idx]

def array_shuffle(elems: Iterable[np.ndarray], random: np.random.RandomState = None) -> List[np.ndarray]:
    """
        shuffles an array of arrays, not in place
        elems: array of np.ndarrays to shuffle
        random: random state to use, uses np.random if not provided
        returns: list with same elements as elems, in shuffled order
    """
    elems = list(elems)
    if random is None:
        random = np.random
    indices = np.arange(len(elems))
    np.random.shuffle(indices)
    return [elems[i] for i in indices]

def optional_random(rand_seed: Union[int, np.random.RandomState] = None):
    if rand_seed is None:
        return np.random
    elif isinstance(rand_seed, int):
        return np.random.RandomState(rand_seed)
    else:
        return rand_seed

PROJECT_BASE_DIR: str = os.path.dirname(__file__)

