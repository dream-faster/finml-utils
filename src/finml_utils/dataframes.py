from collections.abc import Callable, Sequence
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from more_itertools import consecutive_groups
from tqdm import tqdm

from .list import filter_none

TPandas = TypeVar("TPandas", pd.DataFrame, pd.Series)


def __concat_on_axis(axis: str) -> Callable:
    def concat_on(dfs: Sequence[pd.DataFrame | pd.Series | None]) -> pd.DataFrame:
        filtered = filter_none(dfs)
        if len(filtered) == 0:
            return None  # type: ignore
        return pd.concat(filtered, axis=axis).sort_index()

    return concat_on


def concat_on_columns(dfs: Sequence[pd.DataFrame | pd.Series | None]) -> pd.DataFrame:
    return __concat_on_axis("columns")(dfs)


def concat_on_index(dfs: Sequence[pd.DataFrame | pd.Series | None]) -> pd.DataFrame:
    return __concat_on_axis("index")(dfs)


def trim_initial_nans(series: pd.Series) -> pd.Series:
    first_valid_index = series.first_valid_index()
    if first_valid_index is None:
        return pd.Series(dtype="float64")
    return series.loc[first_valid_index:]


def n_of_max_consecutive_nan(series: pd.Series):
    series = trim_initial_nans(series)
    return (
        series.isna()
        .astype(int)
        .groupby(series.notna().astype(int).cumsum())
        .sum()
        .max()
    )


def get_groups_of_nans(series: pd.Series) -> pd.DataFrame:
    series = trim_initial_nans(series)

    series = series.reset_index(drop=True)
    m = series[series.isna()]
    groups = [list(i) for i in consecutive_groups(m.index)]
    d = {ele: e for e, item in enumerate(groups) for ele in item}
    return (
        pd.Series(m.index, index=m.index.map(d))
        .groupby(level=0)
        .agg(["min", "max", "count"])
        .sort_values("count", ascending=False)
    )


def remove_before_nan_gap(
    series: pd.Series,
    larger_than: int,
    verbose: bool = False,
) -> pd.Series | None:
    series = trim_initial_nans(series)
    groups = get_groups_of_nans(series)
    if groups.empty:
        return series
    if groups.iloc[0]["count"] > larger_than:
        purged_series = series.iloc[groups.iloc[0]["max"] :]
        if verbose:
            print(
                f"Only keeping last part of series: {series.name}",
                f"new length: {len(purged_series)}",
            )
        if len(purged_series) < 2:
            return None
        return purged_series

    return series


def concat_on_index_without_duplicates(
    series: Sequence[TPandas], keep: Literal["first", "last"] = "last"
) -> TPandas:
    if len(series) == 0:
        return pd.DataFrame()
    if len(series) == 1:
        return series[0]

    if len(series) > 2:
        keep_this = series[0] if keep == "first" in keep else series[-1]
        concatenated = pd.concat(
            series[1:] if keep == "first" else series[:-1], axis="index"
        )
        _first = concatenated.index.duplicated(
            keep="first" if keep == "last" else "last"
        )
        _last = concatenated.index.duplicated(keep=keep)
        concatenated = concatenated[~_last].fillna(concatenated[~_first])
        concatenated = keep_this.reindex(
            keep_this.index.union(concatenated.index)
        ).fillna(concatenated)
    else:
        concatenated = pd.concat(series, axis="index")
        _first = concatenated.index.duplicated(
            keep="first" if keep == "last" else "last"
        )
        _last = concatenated.index.duplicated(keep=keep)

        concatenated = concatenated[~_last].fillna(concatenated[~_first])

    concatenated = concatenated.sort_index()

    if isinstance(series, pd.Series) and isinstance(concatenated, pd.DataFrame):
        return concatenated.squeeze()
    return concatenated


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate columns from a DataFrame, keep the last occurences.
    """
    return df.loc[:, ~df.columns[::-1].duplicated()[::-1]]  # type: ignore


def remove_columns_with_missing_values(
    df: pd.DataFrame, threshold: float | int
) -> pd.DataFrame:
    def calculate_window_size(window_size: int | float, length: int) -> int:
        return window_size if window_size > 1 else int(length * window_size)  # type: ignore

    notna = df.notna().sum()
    threshold = calculate_window_size(threshold, df.shape[0])
    return df[notna[notna > threshold].index]


def rebase(prices: pd.Series, base: float = 1.0) -> pd.Series:
    """
    Rebase all series to a given intial base.
    This makes comparing/plotting different series together easier.
    Args:
        * prices: Expects a price series/dataframe
        * base (number): starting value for all series.
    """
    return prices.dropna() / prices.dropna().iloc[0] * base


def get_delisted_columns(data: pd.DataFrame, timeframe: int) -> list[str]:
    isnan_period = data.iloc[-timeframe:].isna().sum().sort_values(ascending=False)
    is_discountinued = isnan_period == len(data.iloc[-timeframe:])
    return is_discountinued[is_discountinued].index.to_list()


def adjust_with_nan_ratio(scores: pd.Series, X: pd.DataFrame) -> pd.Series:
    na_ratio = 1 - (X.isna().sum() / len(X))
    return scores * na_ratio


TPandas1 = TypeVar("TPandas1", pd.DataFrame, pd.Series)
TPandas2 = TypeVar("TPandas2", pd.DataFrame, pd.Series)


def add_last_day_if_different(
    lhs: TPandas1, rhs: TPandas2
) -> tuple[TPandas1, TPandas2]:
    lhs, rhs = lhs.copy(), rhs.copy()
    last_day = max(lhs.index[-1], rhs.index[-1])
    if lhs.index[-1] != last_day:
        lhs.loc[last_day] = lhs.iloc[-1]
    if rhs.index[-1] != last_day:
        rhs.loc[last_day] = rhs.iloc[-1]
    return lhs, rhs


def ffill_last_day(series: pd.Series) -> pd.Series:
    if not series.isna().iloc[-1]:
        return series
    series = series.copy()
    series.iloc[-1] = series.loc[series.last_valid_index()]
    return series


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.nunique() > 1]


def apply_function_batched(
    df: pd.DataFrame,
    func: Callable,
    batch_columns: int | None,
    display_progress: bool = False,
) -> pd.DataFrame:
    if batch_columns is not None:
        batches = [
            func(df.iloc[:, i : i + batch_columns])
            for i in tqdm(
                range(0, len(df.columns), batch_columns), disable=not display_progress
            )
        ]
        return concat_on_columns(batches)
    return func(df)


def print_full(x: pd.DataFrame) -> None:
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:20,.2f}".format)
    pd.set_option("display.max_colwidth", None)
    print(x)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")


def replace_inf_with_nan(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)
