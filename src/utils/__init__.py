import os
import getpass

USERNAME = getpass.getuser()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def abspath(*path):
    """
    return absolute path from ROOT_DIR
    """
    abs_path = os.path.join(ROOT_DIR, *path)
    # file
    if '.' in path[-1]:
        dir_path = os.path.dirname(abs_path)
    else:
        dir_path = abs_path

    exists = os.path.exists(dir_path)
    if not exists:
        os.makedirs(dir_path)

    return abs_path