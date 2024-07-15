from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", pd.DataFrame, pd.Series)


def zscore(
    window: int | None,
    min_periods: int,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
    absolute: bool = False,
) -> Callable[[T], T]:
    def zscore_(df: T) -> T:
        r = (
            df.expanding(min_periods)
            if window is None
            else df.rolling(window=window, min_periods=min_periods)
        )
        m = r.mean().shift(1).astype(np.float32)
        s = r.std(ddof=0).shift(1).add(1e-5).astype(np.float32)
        output = (df - m).div(s)
        if clip_lower is not None or clip_upper is not None:
            output = output.clip(lower=clip_lower, upper=clip_upper)
        if absolute:
            return output.abs()
        return output

    zscore_.__name__ = f"zscore_{window}"
    return zscore_
