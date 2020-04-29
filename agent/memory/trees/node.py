from typing import TypeVar, Generic, List

T = TypeVar("T")

class Node(Generic[T]):
    relations: List[str] = ['left', 'right', 'parent'] #possible links to other nodes

    def __init__(self, value: T, parent: Node[T] = None):
        self.value: T = value
        self.depth: int = parent.depth + 1 if parent is not None else 0
        self.size: int = 1

        self.parent: Node[T] = parent
        self.left: Node[T] = None 
        self.right: Node[T] = None

    @staticmethod
    def _is_valid_relation(attr: str) -> bool:
        return attr in Node.relations

    def add_relation(self, value: T, attr: str) -> Node[T]:
        assert Node._is_valid_relation(attr)
        assert not self._has_relation_(attr)
        setattr(self, attr, Node(value, self))
        return self._get_relation_(attr)

    def has_relation(self, attr: str) -> Node[T]:
        assert Node._is_valid_relation(attr)
        return self._get_relation_(attr) is None

    def get_relation(self, attr: str) -> Node[T]:
        assert Node._is_valid_relation(attr)
        return getattr(self, attr)