import numpy as np
import pandas as pd
from finml_utils import (
    center,
    fill_na_with_mean,
    fill_na_with_noise,
    fill_with,
    pearson_corr_pandas,
)


def test_center():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_centered = center(X)
    assert np.allclose(X_centered, np.array([[-2, -2], [0, 0], [2, 2]]))


def test_fill_na_with_mean():
    X = np.array([[1, np.nan], [3, 4], [np.nan, 6]])
    X_filled = fill_na_with_mean(X)
    assert np.allclose(X_filled, np.array([[1, 5], [3, 4], [2, 6]]))


def test_fill_na_with_noise():
    X = np.array([[1, np.nan], [3, 4], [np.nan, 6]])
    X_filled = fill_na_with_noise(X)
    assert not np.isnan(X_filled).any()


def test_fill_with():
    X = np.array([[1, np.nan], [3, 4], [np.nan, 6]])
    X_filled_with_mean = fill_with(X, "mean")
    X_filled_with_noise = fill_with(X, "noise")
    assert np.allclose(X_filled_with_mean, np.array([[1, 5], [3, 4], [2, 6]]))
    assert not np.allclose(X_filled_with_mean, X_filled_with_noise)


def test_pearson():
    num_cols = 17
    num_rows = 200
    df = pd.DataFrame(np.random.rand(num_rows, num_cols), columns=range(num_cols))
    corr = pearson_corr_pandas(df, "mean")

    assert corr.shape == (num_cols, num_cols)
    assert corr.isna().sum().sum() == 0
