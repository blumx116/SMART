import os
from typing import List, Iterable, TypeVar, Union, Tuple, Generic

import numpy as np
from np.random import RandomState

T = TypeVar("T")
V = TypeVar("V")

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

def optional_random(rand_seed: Union[int, RandomState] = None):
    if rand_seed is None:
        return np.random
    elif isinstance(rand_seed, int):
        return RandomState(rand_seed)
    else:
        return rand_seed

def array_unique(data: Iterable[np.ndarray], return_inverse=False, 
        return_counts = False) -> List[np.ndarray]:
    """
        used for essentially calling np.unique on a list of np.ndarrays
        NOTE: this is only needed because the default np.unique function can't handle arrays of arrays
        Parameters
        ----------
        data: Iterable[np.ndarray] : n_data => [ndarray_dims]
            array of np.ndarrays to find unique elements from
        return_inverse: bool
            see np.unique docs
        return_counts: bool
            see np.unique docs
        Returns
        -------
        uniquas : List[np.ndarray] : n_unique => [ndarray_dims]
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



def array_equal(el1: T, el2: T) -> bool:
    """
        Checks if element 1 and 2 are equal
        NOTE: this only exists because numpy's default handling of the == operator
        is different from that of normal objects. This provides a data agnostic way
        to check
        Parameters
        ----------
        el1: T
            left hand side of equality
        el2: T
            right hand side of equality
        Returns
        -------
        equal?: bool
            whether or not el1 == el2
    """
    if isinstance(el1, np.ndarray) and isinstance(el2, np.ndarray):
        return np.array_equal(el1, el2)
    elif isinstance(el1, Iterable) and isinstance(el2, Iterable):
        if len(el1) != len(el2):
            return False 
        if len(el1) == len(el2) == 0:
            return True 
        else:
            return array_equal(el1[0], el2[0]) and array_equal(el1[1:], el2[1:])
    else:
        return el1 == el2

def array_contains(el: T, elems: Iterable[T]) -> bool:
    """
        returns whether or not el is in elems
        NOTE: This only exists because the default contains method doesn't work with np.ndarrays
        Parameters
        ----------
        el: T
            element to check for in elems
        elems: Iterable[T]
            elements to check whether el is a part of
        Returns
        -------
        present: bool
            whether or not el was present in elems
        
    """
    for element in list:
        if array_equal(el, element):
            return True
    return False

def array_random_choice(elems: Iterable[np.ndarray], random: Union[int, RandomState] = None) -> np.ndarray:
    """
        Randomly selects an element from elems to return. 
        NOTE: Exists because default numpy function not compatible with np.ndarrays
        Parameters
        ----------
        elems : Iterable[np.ndarray]
            elements to choose from among
        random : Union[int, RandomState]
            random seed to use, uses global random if none provided
        Returns
        -------
        selected: np.ndarray
            the randomly selected element from the list
    """
    random: RandomState = optional_random(random)
    elems = list(elems)
    if random is None:
        random = np.random
    idx = random.choice(len(elems))
    return elems[idx]



def array_shuffle(elems: Iterable[np.ndarray], random: Union[int, RandomState] = None) -> List[np.ndarray]:
    """
        shuffles an array of arrays, out of place
        NOTE: Exists because default numpy function not compatible with np.ndarrays
        elems: array of np.ndarrays to shuffle
        Parameters
        ----------
        elems: Iterable[np.ndarray]
            elements to be shuffled, not in place. Technically works on even if 
            elements are not np.ndarrays
        random: Union[int, RandomState]
            random seed to use, uses global random if none provided
        Returns
        -------
        shuffled: List[np.ndarray]
            list with same elements as elems, in shuffled order
    """
    random = optional_random(random)
    elems = list(elems)
    if random is None:
        random = np.random
    indices = np.arange(len(elems))
    np.random.shuffle(indices)
    return [elems[i] for i in indices]

def bool_random_choice(probability: float, rand_seed: Union[int, RandomState] = None) -> bool:
    """
        Randomly chooses True with probability equal to 'probability'
        and False otherwise, using rand_seed for random number generator
        Possibly redundant.
        Parameters
        ----------
        probability: float in [0, 1]
            probability of returning True
        rand_seed: Union[int, RandomState]
            random seed to use, uses global random if none provided
        Returns
        -------
        choice: bool
            True or False, chosen randomly
        """"
    assert 0 <= probability <= 1
    random: RandomState = optional_random(rand_seed)
    return random.uniform() < probability

PROJECT_BASE_DIR: str = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir))

class NumPyDict(Generic[T, V]):
    """
        A replacement class for dictionaries when numpy npdarrays need to be used as keys
        Works by hashing key to bytes if np.ndarray, and behaving like a normal dict otherwise
    """
    def __init__(self, dtype=float):
        self.inner = {}
        self.dtype = dtype

    def _pack_np(self, key: T) -> T:
        """
            converts np.ndarrays to bytes
            everything else unchanged
        """
        if isinstance(key, np.ndarray):
            return key.tobytes()
        return key

    def _unpack_np(self, key: T) -> T:
        """
            convertes bytes to np.ndarray
            everything else unchanged
        """
        if isinstance(key, bytes):
            return np.frombuffer(key, dtype=self.dtype)
        return key

    def __getitem__(self, key: T) -> V:
        return self.inner[self._pack_np(key)]

    def __setitem__(self, key: T, value: V) -> None:
        self.inner[self._pack_np(key)] = value

    def __iter__(self):
        return iter(self.inner)

    def __next__(self):
        return self.inner.__next__()

    def __contains__(self, key: T) -> bool:
        return self._pack_np(key) in self.inner 

    def keys(self) -> Iterable[T]:
        return list(map(self._unpack_np, self.inner.keys()))

    def values(self) -> Iterable[V]:
        return self.inner.values()
