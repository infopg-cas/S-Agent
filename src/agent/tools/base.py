from typing import Callable

class Tool:
    def __init__(self, name: str, func: Callable) -> None:
        self.name = name
        self.func = func

    def act(self, **kwargs) -> str:
        return self.func(**kwargs)