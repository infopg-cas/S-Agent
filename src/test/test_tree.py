from src.utils.tree_structure import Tree

if __name__ == "__main__":
    import pprint
    tree = Tree()
    tree.add_root({"order1": 1, "content": "root1"})
    tree.add_node("order1", 1, {"order1": 2, "content": "child1"})
    tree.add_node("order1", 2, {"order1": 3, "content": "child2"})
    tree.add_root({"order": 1, "content": "root2"})
    pprint.pprint(tree.print_tree())