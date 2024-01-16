from typing import TypeVar

import pandas as pd
from more_itertools import consecutive_groups

TPandas = TypeVar("TPandas", pd.DataFrame, pd.Series)


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
    series: list[TPandas],
    keep: str = "last",
    vectorized: str = ["original", "vectorized", "combine_first"],
) -> TPandas:
    if len(series) == 0:
        return pd.DataFrame()
    if len(series) == 1:
        return series[0]

    # non vectorized
    if vectorized == "combine_first":
        series = series[::-1] if keep == "last" else series
        concatenated = series[0]
        for s in series[1:]:
            concatenated = concatenated.combine_first(s)

    # vectorized
    elif vectorized == "vectorized":
        concatenated = pd.concat(series, axis=0).groupby(level=0).first()
    elif vectorized == "vectorized_fast":
        concatenated = pd.concat(series, axis="index")
        _first = concatenated.index.duplicated(keep="first")
        _last = concatenated.index.duplicated(keep="last")

        concatenated = concatenated[_last].fillna(concatenated[_first])
    elif vectorized == "vectorized_fast_2":
        concatenated = pd.concat(series, axis="index")
        _last = concatenated.index.duplicated(keep="last")

        concatenated = concatenated[_last].fillna(concatenated[~_last])

    # original
    else:
        concatenated = pd.concat(series, axis="index")
        concatenated = concatenated[~concatenated.index.duplicated(keep=keep)]

    if isinstance(series, pd.Series) and isinstance(concatenated, pd.DataFrame):
        return concatenated.squeeze()
    return concatenated
