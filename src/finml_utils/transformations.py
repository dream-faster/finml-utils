from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", pd.DataFrame, pd.Series)


def zscore(window: int | None, min_periods: int) -> Callable[[T], T]:
    def zscore_(df: T) -> T:
        r = (
            df.expanding(min_periods)
            if window is None
            else df.rolling(window=window, min_periods=min_periods)
        )
        m = r.mean().shift(1).astype(np.float32)
        s = r.std(ddof=0).shift(1).add(1e-5).astype(np.float32)
        return (df - m) / s

    zscore_.__name__ = f"zscore_{window}"
    return zscore_
