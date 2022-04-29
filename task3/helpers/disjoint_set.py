from typing import List, Tuple, Dict
from pprint import pprint


class Node(object):
    def __init__(self, val):
        self.val = val
        self.rank = 0
        self.parent = self
        self.child = None

    def __repr__(self):
        return f"Node({self.val}, {self.rank}, {self.parent.val})"

    def __eq__(self, other):
        return self.val == other.val


def disjoint_set(paths: List) -> Tuple[Dict[str, str], Dict[str, List]]:
    nodes = {}
    for anchor_str, pos_str, _ in paths:
        # init for each node
        if anchor_str not in nodes:
            nodes[anchor_str] = Node(anchor_str)
        if pos_str not in nodes:
            nodes[pos_str] = Node(pos_str)

        anchor = nodes[anchor_str]
        pos = nodes[pos_str]
        union_(anchor, pos)

    path2root = {key: find_set_(node).val for key, node in nodes.items()}
    root2set = {}
    for key, root in path2root.items():
        if root not in root2set:
            root2set[root] = []
        root2set[root].append(key)

    return path2root, root2set


def find_set_recursive_(node: Node):
    # with path compression
    if node.parent == node:
        return node

    parent = find_set_(node.parent)
    node.parent = parent

    return parent


def find_set_(node: Node):
    node_iter = node
    while node_iter.parent != node_iter:
        # print("in the upward loop")
        node_iter.parent.child = node_iter
        node_iter = node_iter.parent
    root = node_iter

    node_iter = root
    while node_iter is not None:
        # print("in the downward loop")
        node_iter.parent = root
        node_iter_child = node_iter.child
        node_iter.child = None
        node_iter = node_iter_child

    return root


def union_(node1: Node, node2: Node):
    root1 = find_set_(node1)
    root2 = find_set_(node2)
    if root1.rank > root2.rank:
        root2.parent = root1
    elif root2.rank > root1.rank:
        root1.parent = root2
    else:
        root2.parent = root1
        root1.rank += 1


def check_key_in_set(root2set: Dict):
    for key, val in root2set.items():
        try:
            assert key in val
        except AssertionError:
            print(key)
            print(val)
            raise AssertionError


if __name__ == '__main__':
    # toy example for sanity check: since neg sample is not used, simply set it to -1
    paths = [(1, 2, -1), (2, 3, -1), (4, 1, -1), (5, 6, -1), (7, 5, -1), (8, 6, -1), (9, 10, -1)]
    # expected: [1, 2, 3, 4], [5, 6, 7, 8], [9, 10]
    path2root, root2set = disjoint_set(paths)
    check_key_in_set(root2set)
    pprint(path2root)
    pprint(root2set)
