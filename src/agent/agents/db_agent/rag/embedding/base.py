import os
from abc import ABC
def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )
    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE

def get_home_cache_dir():
    dir = os.path.join(os.path.expanduser('~'), ".ragflow")
    try:
        os.mkdir(dir)
    except OSError as error:
        pass
    return dir

class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list, batch_size=32):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")
