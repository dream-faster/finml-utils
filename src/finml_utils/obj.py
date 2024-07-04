from copy import deepcopy
from typing import TypeVar

T = TypeVar("T")


def copy_set_properties(obj: T, **kwargs) -> T:
    obj = deepcopy(obj)
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj
