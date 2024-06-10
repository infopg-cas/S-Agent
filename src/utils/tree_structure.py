from collections import deque

class TreeNode:
    def __init__(self, attributes):
        # Dynamically set attributes based on dictionary keys
        for key, value in attributes.items():
            setattr(self, key, value)
        self.children = []

    def add_child(self, child_node):
        """Add a child node to this node."""
        self.children.append(child_node)

    def __repr__(self):
        """Representation of the node, showing its attributes and children."""
        attributes = vars(self).copy()
        attributes['children'] = [repr(child) for child in self.children]
        return str(attributes)

    def to_dict(self):
        """Return a dictionary representation of the node, including its attributes and children."""
        attributes = vars(self).copy()  # Copy attributes to include in the dictionary
        attributes['children'] = [child.to_dict() for child in self.children]  # Recursively convert children
        return attributes


class Tree:
    def __init__(self):
        self.roots = []

    def add_root(self, attributes):
        """Add a new root to the tree."""
        new_root = TreeNode(attributes)
        self.roots.append(new_root)
        return new_root

    def add_node(self, parent_attribute, parent_value, child_attributes):
        for root in self.roots:
            parent_node = self.find_node_by_attribute(root, parent_attribute, parent_value)
            if parent_node is not None:
                parent_node.add_child(TreeNode(child_attributes))
                return
        self.add_root(child_attributes)

    # BSF
    def find_node_by_attribute(self, current_node, attribute, value):
        queue = deque([current_node])
        while queue:
            node = queue.popleft()
            if attribute in vars(node) and vars(node)[attribute] == value:
                return node
            for child in node.children:
                queue.append(child)
        return None

    def find_node(self, attribute, value):
        for root in self.roots:
            node = self.find_node_by_attribute(root, attribute, value)
            if node:
                return node
        return None

    def print_tree(self, start_node=None):
        return [root.to_dict() for root in self.roots]

    def sum_children_scores(self, node):
        if not node.children:
            return node.score

        total_score = 0
        for child in node.children:
            total_score += self.sum_children_scores(child)

        node.score = total_score
        return node.score


if __name__ == "__main__":
    # Example usage
    import pprint
    tree = Tree()
    tree.add_root({"order1": 1, "content": "root1"})
    tree.add_node("order1", 1, {"order1": 2, "content": "child1"})
    tree.add_node("order1", 2, {"order1": 3, "content": "child2"})
    tree.add_root({"order": 1, "content": "root2"})
    pprint.pprint(tree.print_tree())