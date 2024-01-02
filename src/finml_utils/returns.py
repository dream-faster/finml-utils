from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)


def to_log_returns(data: T) -> T:
    return np.log1p(to_returns(data))


def to_returns(data: T) -> T:
    if isinstance(data, pd.DataFrame):
        return data.apply(to_returns)
    if data.min() < 0:
        return data.diff() / data.shift(1).abs()
    return data / data.shift(1) - 1
