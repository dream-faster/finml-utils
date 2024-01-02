from __future__ import annotations

from enum import Enum
from typing import Self


class ParsableEnum(Enum):
    @classmethod
    def from_str(cls, value: str | ParsableEnum, safe: bool = False) -> Self:
        if isinstance(value, cls):
            return value
        for item in cls:
            if item.name == value:
                return item  # type: ignore
        if safe:
            return None  # type: ignore
        raise ValueError(f"Unknown {cls.__name__}: {value}")
