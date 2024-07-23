import pandas as pd

from .list import flatten, unique


def get_column_names(column_pattern: str, X: pd.DataFrame) -> pd.Index | list[str]:
    if column_pattern == "all":
        return X.columns
    if column_pattern.endswith("*"):
        to_match = column_pattern.split("*")[0]
        return [col for col in X.columns if col.startswith(to_match)]
    if column_pattern.startswith("*"):
        to_match = column_pattern.split("*")[1]
        return [col for col in X.columns if col.endswith(to_match)]
    return [column_pattern]


def get_list_column_names(columns: list[str], X: pd.DataFrame) -> list[str] | pd.Index:
    assert isinstance(columns, list)
    if len(columns) == 1:
        return get_column_names(columns[0], X)
    return unique(flatten([get_column_names(c, X) for c in columns]))
