from typing import TypeVar, Generic, List

T = TypeVar("T")

class BinaryTreeNode(Generic[T]):
    relations: List[str] = ['left', 'right', 'parent'] #possible links to other nodes

    def __init__(self, value: T, parent: BinaryTreeNode[T] = None):
        self.value: T = value
        self.depth: int = parent.depth + 1 if parent is not None else 0

        self.parent: BinaryTreeNode[T] = parent
        self.left: BinaryTreeNode[T] = None 
        self.right: BinaryTreeNode[T] = None

    @staticmethod
    def _is_valid_relation(attr: str) -> bool:
        return attr in BinaryTreeNode.relations

    def add_relation(self, value: T, attr: str) -> BinaryTreeNode[T]:
        assert BinaryTreeNode._is_valid_relation(attr)
        assert not self._has_relation_(attr)
        setattr(self, attr, BinaryTreeNode(value, self))
        return self._get_relation_(attr)

    def has_relation(self, attr: str) -> BinaryTreeNode[T]:
        assert BinaryTreeNode._is_valid_relation(attr)
        return self._get_relation_(attr) is None

    def get_relation(self, attr: str) -> BinaryTreeNode[T]:
        assert BinaryTreeNode._is_valid_relation(attr)
        return getattr(self, attr) is None