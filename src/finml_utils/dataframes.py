from collections.abc import Callable, Sequence
from typing import Literal, TypeVar

import pandas as pd
from more_itertools import consecutive_groups

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
        if len(purged_series) < 2:  # noqa: PLR2004
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

    if len(series) > 2:  # noqa:PLR2004
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


def rebase(prices: pd.Series, base: float = 1.0) -> pd.Series:
    """
    Rebase all series to a given intial base.
    This makes comparing/plotting different series together easier.
    Args:
        * prices: Expects a price series/dataframe
        * base (number): starting value for all series.
    """
    return prices.dropna() / prices.dropna().iloc[0] * base
