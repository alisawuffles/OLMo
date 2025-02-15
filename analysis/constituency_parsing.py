import re


class Node:
    def __init__(self, label, parent=None):
        self.label = label
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


def tokenize(s):
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    return s.split()


def parse_constituency_tree(tokens):
    stack = []
    root = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "(":
            i += 1
            label = tokens[i]
            new_node = Node(label)
            if stack:
                stack[-1].add_child(new_node)
            else:
                root = new_node
            stack.append(new_node)
            i += 1
        elif token == ")":
            if stack:
                stack.pop()
            i += 1
        else:
            word = token
            word_node = Node(word)
            if stack:
                stack[-1].add_child(word_node)
            else:
                raise ValueError("Word outside of any non-terminal")
            i += 1
    return root


def collect_words(node, word_list):
    if not node.children:
        word_list.append(node)
    else:
        for child in node.children:
            collect_words(child, word_list)


def get_path(node):
    path = []
    current = node.parent
    while current is not None:
        path.append(current)
        current = current.parent
    return path


def find_lca(path1, path2):
    path2_nodes = set(path2)
    for node in path1:
        if node in path2_nodes:
            return node
    return None


def get_min_levels(tree_str, idx1, idx2):
    tokens = tokenize(tree_str)
    root = parse_constituency_tree(tokens)
    word_list = []
    collect_words(root, word_list)
    assert idx1 < len(word_list) and idx2 < len(word_list), f"{idx1} or {idx2} is too large"

    node1 = word_list[idx1]
    node2 = word_list[idx2]

    path1 = get_path(node1)
    path2 = get_path(node2)

    lca = find_lca(path1, path2)

    if not lca:
        return -1

    index1 = path1.index(lca)
    index2 = path2.index(lca)

    return max(index1, index2)
