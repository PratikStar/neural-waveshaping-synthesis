import os, sys
from typing import Callable, Sequence


def apply(fn: Callable[[any], any], x: Sequence[any]):
    if type(x) not in (tuple, list):
        raise TypeError("x must be a tuple or list.")
    return type(x)([fn(element) for element in x])


def apply_unpack(fn: Callable[[any], any], x: Sequence[Sequence[any]]):
    if type(x) not in (tuple, list):
        raise TypeError("x must be a tuple or list.")
    return type(x)([fn(*element) for element in x])


def unzip(x: Sequence[any]):
    return list(zip(*x))


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
