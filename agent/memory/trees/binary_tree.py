from typing import TypeVar, Generic, List

from agent.memory.trees.binary_tree_node import BinaryTreeNode

T = TypeVar("T")

class BinaryTree(Generic[T]):
    """
        Like a binary search tree, but the values are not necessarily intrinsically
        ordered. Instead, elements are inserted directly before or after existing
        elements, creating an implicit ordering
    """
    def __init__(self):
        self.root: BinaryTreeNode[T] = BinaryTreeNode(None, None)
        self.size = 0 #don't count root node

    def _opposite_direction_(self, direction: str) -> str:
        assert direction in ['left', 'right']
        return'left' if direction == 'right' else 'right'

    def list_nodes(self) -> List[BinaryTreeNode[T]]:
        result = []
        if len(self) == 0:
            return []
        else:
            result

    def add_left(self, value: T, node: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Adds a new node with value T immediately to the left of node 'node'
            value : the value stored at the new node
            node: the immediately node to the right of the new one in ordering
            returns : the new node
        """
        return self._add_direction_(value, 'left', node)

    def add_right(self, value: T, node: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Adds a new node with value T immediately to the right of node 'node'
            value : the value stored at the new node
            node: the node immediately to the left of the new one in ordering
            returns : the new node
        """
        return self._add_direction_(value, 'right', node)
        
    def get_next_left(self,  node: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Gets the rightmost node that is left of the node 'node'
            If these were numbers, then it would be the greatest number 
            less than node.value
            node: the node immediately to the right of the returned node
            returns : the next value to the left, root if not found
        """
        return self._get_next_in_direction_('left', node)

    def get_next_right(self,  node: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Gets the lefttmost node that is right of the node 'node'
            If these were numbers, then it would be the lowest number 
            greater than node.value
            node: the node immediately to the left of the returned node
            returns : the next value to the right, root if not found
        """
        return self._get_next_in_direction_('left', node)

    def get_leftmost(self, subtree_root: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Gets the leftmost node in the tree rooted at subtree_root.
            May return subtree_root itself. If subtree_root not provided,
            searches entire tree.
            subtree_root: the root of the subtree to search
            returns : the leftmost node
        """
        return self._get_furthest_in_direction('left', subtree_root)

    def get_rightmost(self, subtree_root: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Gets the rightmost node in the tree rooted at subtree_root.
            May return subtree_root itself. If subtree_root not provided,
            searches entire tree.
            subtree_root: the root of the subtree to search
            returns : the rightmost node
        """
        return self._get_furthest_in_direction('right', subtree_root)


    def _add_direction_(self, value: T, direction: str, node: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Adds the value in a new node that is directly to the left or right of parent
            for instance, after calling _add_direction_(direction='left'), the new node will
            be the rightmost value to the left of 'node', not necessarily a direct child
            value: T, the new value to add to the tree
            direction: left or right, where the new value is in relation to 'node'
            'node': the 'node' to add to the left or right of, doesn't need to have free spots
            returns : the newly created node
        """
        self.size += 1
        other: str = self._opposite_direction_(direction)
        if not node.has_relation(direction): #if you can add directly to left or right, do so
            return node.add_relation(value, direction)
        else:
            node = node.get_relation(direction) #otherwise, go one over
            while node.has_relation(other): #go as far as you can back
                node = node.get_relation(other)
            return node.add_relation(value, other) #and add

    def _get_next_in_direction_(self, direction: str, node: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Gets the next node over in the tree
            For instance, if this were a binary search tree of numbers,
            and _get_direction_('left') were called, it would return the node
            corresponding to the highest number smaller than 'node'.
            If not found, root is returned
            direction : ['left', 'right'] : which direction to search in
            node: the node to search from the left or right of
            returns : the next node found after searching in that direction
        """
        other: str = self._opposite_direction_(direction)

        #Case 1: Search down the tree
        #Example : direction = left, we want the rightmost element of the
        # left subtree
        if node.has_relation(direction):
            node = node.get_relation(direction)
            while node.has_relation(other):
                node = node.get_relation(other)
            return node 

        #Case 2: Search up the tree
        #Example: direction = left, we want the first ancestor
        #Where we are in the right subtree of that ancestor
        else:
            while node != self.root:
                parent: BinaryTreeNode[T] = node.get_relation('parent')
                if parent.get_relation(other) == node:
                    return parent
                else:
                    node = parent 

    def _get_furthest_in_direction(self, direction: str, subtree_root: BinaryTreeNode[T]) -> BinaryTreeNode[T]:
        """
            Gets the node that is furthest to in the given direction 
            in the subtree rooted at 'subtree_root'. May return subtree_root
            itself if necessary. If subtree_root not provided, uses
            the root of the entire tree.
            direction : ['left', 'right'], the direction to search in
            subtree_root: the root of the subtree to search
            returns : the node that is furthest in the given direction
        """
        while subtree_root.has_relation(direction):
            subtree_root = subtree_root.get_relation(direction)
        return subtree_root

    def __len__(self) -> int:
        return self.size