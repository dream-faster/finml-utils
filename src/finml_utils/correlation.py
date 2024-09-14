from typing import Literal

import numpy as np
import pandas as pd


def pearson_corr_pandas(
    df: pd.DataFrame,
    fill_na: Literal["no", "mean", "noise", "zero"],
) -> pd.DataFrame:
    return pd.DataFrame(
        pearson_corr_numpy(df.values, fill_na=fill_na),
        index=df.columns,
        columns=df.columns,
    )


def pearson_corr_numpy(
    df: np.ndarray,
    fill_na: Literal["no", "mean", "noise", "zero"],
) -> np.ndarray:
    filled = fill_with(df, fill_na)
    return np.corrcoef(center(filled), rowvar=False, dtype=np.float32)


def center(X: np.ndarray) -> np.ndarray:
    return X - np.nanmean(X, axis=0)


def fill_with(
    X: np.ndarray, fill_na: Literal["no", "mean", "noise", "zero"]
) -> np.ndarray:
    if fill_na == "no":
        return X
    if fill_na == "mean":
        return fill_na_with_mean(X)
    if fill_na == "noise":
        return fill_na_with_noise(X)
    if fill_na == "zero":
        return fill_na_with_zero(X)
    raise ValueError(f"fill_na={fill_na} is not supported")


def fill_na_with_mean(X: np.ndarray) -> np.ndarray:
    return np.where(
        np.isnan(X),
        np.nanmean(X, axis=0),
        X,
    )


def fill_na_with_zero(X: np.ndarray) -> np.ndarray:
    return np.where(
        np.isnan(X),
        0.0,
        X,
    )


def fill_na_with_noise(X: np.ndarray) -> np.ndarray:
    return np.where(
        np.isnan(X),
        np.random.uniform(
            low=np.nanmin(X),
            high=np.nanmax(X),
            size=X.shape,
        ),
        X,
    )
