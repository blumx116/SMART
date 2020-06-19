from typing import TypeVar, Generic, Iterable

import numpy as np

T = TypeVar("T")
V = TypeVar("V")

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