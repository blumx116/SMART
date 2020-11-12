from typing import TypeVar, Generic, List, Optional

from .node import Node

T = TypeVar("T")
V = TypeVar("V")

class Tree(Generic[T]):
    """
        Like a binary search tree, but the values are not necessarily intrinsically
        ordered. Instead, elements are inserted directly before or after existing
        elements, creating an implicit ordering
    """

    @staticmethod
    def add_left(value: T, node: Node[T]) -> Node[T]:
        """
            Adds a new node with value T immediately to the left of node 'node'
            value : the value stored at the new node
            node: the immediately node to the right of the new one in ordering
            returns : the new node
        """
        return Tree._add_direction_(value, 'left', node)

    @staticmethod
    def add_right(value: T, node: Node[T]) -> Node[T]:
        """
            Adds a new node with value T immediately to the right of node 'node'
            value : the value stored at the new node
            node: the node immediately to the left of the new one in ordering
            returns : the new node
        """
        return Tree._add_direction_(value, 'right', node)

        
    @staticmethod
    def get_by_index(node: Node[T], index: int) -> Node[T]:
        """
            Returns the index'th item in the tree. Same result as if 
            you were to flatten the tree to a list and access by index
            root: root of the tree to be queried
            index: int, index to be queried for. Support negative values
            returns: the node found
        """
        if index < 0:
            index = node.size + index
        if index > node.size or index < 0:
            raise IndexError(f"tree index {index} out of range")
        left_subtree_size: int = Tree.subtree_size(node, 'left')
        if index < left_subtree_size:
            return Tree.get_by_index(node.get_relation('left'), index)
        elif index == left_subtree_size:
            return node
        else:
            assert node.has_relation('right')
            return Tree.get_by_index(node.get_relation('right'), index - (left_subtree_size + 1))

    @staticmethod
    def get_index_of(node: Node[T], root: Node[T] = None) -> int:
        index: int = Tree.subtree_size(node, 'left')
        while node.has_relation('parent') and node != root:
            if Tree.is_right_child(node):
                index += 1 + Tree.subtree_size(node.get_relation('parent'), 'left')
            node = node.get_relation('parent')
        return index 

    @staticmethod
    def get_leftmost(subtree_root: Node[T]) -> Node[T]:
        """
            Gets the leftmost node in the tree rooted at subtree_root.
            May return subtree_root itself. If subtree_root not provided,
            searches entire tree.
            subtree_root: the root of the subtree to search
            returns : the leftmost node
        """
        return Tree._get_furthest_in_direction('left', subtree_root)

    @staticmethod
    def get_next_left(node: Node[T]) -> Optional[Node[T]]:
        """
            Gets the rightmost node that is left of the node 'node'
            If these were numbers, then it would be the greatest number 
            less than node.value
            node: the node immediately to the right of the returned node
            returns : the next value to the left, root if not found
        """
        return Tree._get_next_in_direction_('left', node)

    @staticmethod
    def get_next_left_parent(node: Node[T]) -> Optional[Node[T]]:
        """
        Gets the ancestor in 'node's ancestor tree that is the rightmost
        but not further right than 'node'.
        Alternately put, it gets the next left node in the tree ignoring node's
        children
        :param node: the reference node to search near
        :return: the found ancestor node, None if all ancestors are left-children
        """
        return Tree._get_next_parent_in_direction_('left', node)

    @staticmethod
    def get_next_right(node: Node[T]) -> Optional[Node[T]]:
        """
            Gets the lefttmost node that is right of the node 'node'
            If these were numbers, then it would be the lowest number 
            greater than node.value
            node: the node immediately to the left of the returned node
            returns : the next value to the right, root if not found
        """
        return Tree._get_next_in_direction_('right', node)


    def get_next_right_parent(node: Node[T]) -> Optional[Node[T]]:
        """
        Gets the ancestor in 'node's ancestor tree that is the leftmost
        but not further left than 'node'.
        Alternately put, it gets the next right node in the tree ignoring node's
        children
        :param node: the reference node to search near
        :return: the found ancestor node, None if all ancestors are right-children
        """
        return Tree._get_next_parent_in_direction_('right', node)

    @staticmethod
    def get_rightmost(subtree_root: Node[T]) -> Node[T]:
        """
            Gets the rightmost node in the tree rooted at subtree_root.
            May return subtree_root itself. If subtree_root not provided,
            searches entire tree.
            subtree_root: the root of the subtree to search
            returns : the rightmost node
        """
        return Tree._get_furthest_in_direction_('right', subtree_root)

    @staticmethod
    def get_root(node: Node[T]) -> Node[T]:
        """
            Gets the root of the tree containing node 'node'
            Parameters
            ----------
            node: Node[T]
                the reference node in the tree where we're searching for 
                the root
            Returns
            -------
            root: Node[T]
                the root node
        """
        while node.has_relation('parent'):
            node = node.get_relation('parent')
        return node

    @staticmethod
    def is_left_of(self, left: Node[T], right: Node[T]) -> bool:
        return Tree._is_in_direction_of_(left, right, 'left')

    @staticmethod
    def is_right_of(right: Node[T], left: Node[T]) -> bool:
        return Tree._is_in_direction_of_(right, left, 'right')

    @staticmethod
    def is_left_child(node: Node[T]) -> bool:
        return Tree._is_child_in_direction_(node, 'left')

    @staticmethod
    def is_right_child(node: Node[T]) -> bool:
        return Tree._is_child_in_direction_(node, 'right')

    @staticmethod
    def is_ancestor_of(ancestor: Node[T], node: Node[T],  bound: Node[T] = None) -> bool:
        """
            node: Node, the potential child
            ancestor: Node, the potential ancestor
            bound: Node, will not check any ancestors higher than bound
            returns whether or not ancestor is an ancestor of node more recently than bound
            NOTE: a node is it's own ancestor
        """
        while True:
            if node == ancestor:
                return True
            if node == bound or not node.has_relation('parent'):
                return False
            node = node.get_relation('parent')


    @staticmethod
    def list_subtree_nodes(root: Node[T], direction: str) -> List[Node[T]]:
        direction = Tree._parse_direction_(direction)
        return Tree.list_nodes(root.get_relation(direction)) if root.has_relation(direction) else []

    @staticmethod
    def list_nodes(root: Node[T]) -> List[Node[T]]:
        return Tree.list_subtree_nodes(root, 'left') + [root] + Tree.list_subtree_nodes(root, 'right')

    @staticmethod
    def mirror_add(existing_node: Node[T], original_root: Node[T], new_root: Node[V], new_value: V) -> Node[V]:
        assert Tree.is_ancestor_of(original_root, existing_node)
        index: int = Tree.get_index_of(existing_node, original_root)
        assert new_root.size >= index #can be equal, because size of original_root should be > than new_root
        if index == 0:
            parent: Node[V] = Tree.get_leftmost(new_root)
            return Tree.add_left(new_value, parent)
        else:
            parent: Node[V] = Tree.get_by_index(new_root, index-1) #get the node immediately to the left
            return Tree.add_right(new_value, parent)

    @staticmethod
    def mirror_get(existing_node: Node[T], original_root: Node[T], new_root: Node[V]) -> Node[V]:
        assert Tree.is_ancestor_of(original_root, existing_node)
        index: int = Tree.get_index_of(existing_node, original_root)
        assert new_root.size > index 
        return Tree.get_by_index(new_root, index)

    @staticmethod
    def subtree_size(node: Node[T], direction: str) -> int:
        direction = Tree._parse_direction_(direction)
        if node.has_relation(direction):
            return node.get_relation(direction).size
        else:
            return 0

    @staticmethod
    def _add_direction_(value: T, direction: str, node: Node[T]) -> Node[T]:
        """
            Adds the value in a new node that is directly to the left or right of parent
            for instance, after calling _add_direction_(direction='left'), the new node will
            be the rightmost value to the left of 'node', not necessarily a direct child
            value: T, the new value to add to the tree
            direction: left or right, where the new value is in relation to 'node'
            'node': the 'node' to add to the left or right of, doesn't need to have free spots
            returns : the newly created node
        """
        other: str = Tree._opposite_direction_(direction)
        if not node.has_relation(direction): #if you can add directly to left or right, do so
            return node.add_relation(value, direction)
        else:
            node = node.get_relation(direction) #otherwise, go one over
            while node.has_relation(other): #go as far as you can back
                node = node.get_relation(other)
            result: Node[T] = node.add_relation(value, other) #and add
            Tree._propagate_tree_size_(result) #update tree size
            return result

    @staticmethod
    def _get_furthest_in_direction_(direction: str, subtree_root: Node[T]) -> Node[T]:
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

    @staticmethod
    def _get_next_in_direction_(
            direction: str,
            node: Node[T]) -> Optional[Node[T]]:
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
        other: str = Tree._opposite_direction_(direction)

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
            while node != None:
                parent: Node[T] = node.get_relation('parent')
                if parent is not None and parent.get_relation(other) == node:
                    return parent
                else:
                    node = parent
            return None

    @staticmethod
    def _is_child_in_direction_(node: Node[T], direction: str) -> bool:
        return node.has_relation('parent') and \
            node.get_relation('parent').has_relation(direction) and \
            node.get_relation('parent').get_relation(direction) == node

    @staticmethod
    def _is_in_direction_of_(other: Node[T], reference: Node[T], direction: str) -> bool:
        assert Tree.get_root(other) == Tree.get_root(reference)
        direction = Tree._parse_direction_(direction)
        # both nodes in the same tree
        if other == reference:
            return False

        current: Node[T] = reference
        while current is not None:
            current = current.get_relation(direction)
            if current == other:
                return True

        # we got to the end of the tree and didn't find it
        return False

    @staticmethod
    def _opposite_direction_(direction: str) -> str:
        direction = Tree._parse_direction_(direction)
        return 'left' if direction == 'right' else 'right'

    @staticmethod
    def _parse_direction_(direction: str) -> str:
        direction = direction.lower()
        assert direction in ['left', 'right']
        return direction 

    @staticmethod
    def _propagate_tree_size_(node: Node[T]) -> None:
        while node.has_relation('parent'):
            node = node.get_relation('parent')
            node.size += 1

    @staticmethod
    def _get_next_parent_in_direction_(
            direction: str,
            node: Node[T]) -> Optional[Node[T]]:
        other_dir: str = Tree._opposite_direction_(direction)
        if not node.has_relation('parent'):
            return None
        if Tree._is_child_in_direction_(node, other_dir):
            return node.get_relation('parent')
        return Tree._get_next_parent_in_direction_(
                direction,
                node.get_relation('parent'))