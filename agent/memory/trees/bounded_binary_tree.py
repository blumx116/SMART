from typing import TypeVar, Generic, List

from agent.memory.trees import BinaryTree, BinaryTreeNode

T = TypeVar("T")

class BoundedBinaryTree(BinaryTree[T]):
    """
        Same as a normal BinaryTree, except that instead of a root, we have
        a lower_bound and an upper_bound. All nodes in the tree must be 
        'between' these two nodes in ordering. The upper_bound serves as 
        the root of the tree, but both nodes have depth 0
    """
    def __init__(self, lower_bound: T, upper_bound:T):
        super().__init__()
        self.root.value = lower_bound
        self.upper_bound: BinaryTree[T] = self.root
        self.upper_bound.value = upper_bound
        self.lower_bound: BinaryTree[T] = \
            self.upper_bound.add_relation(lower_bound, 'left')

        self.upper_bound.depth = 0
        self.lower_bound.depth = 0


    @override 
    def _add_direction_(self, value, direction, node):
        """
            Adds the value in a new node that is directly to the left or right of parent
            for instance, after calling _add_direction_(direction='left'), the new node will
            be the rightmost value to the left of 'node', not necessarily a direct child
            value: T, the new value to add to the tree
            direction: left or right, where the new value is in relation to 'node'
            'node': the 'node' to add to the left or right of, doesn't need to have free spots
            returns : the newly created node
        """
        if direction == 'left': assert node != self.lower_bound
        if direction == 'right': assert node != self.upper_bound
        return super()._add_direction_(value, direction, node)