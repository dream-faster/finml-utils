import numpy as np
import pandas as pd

from src.finml_utils.stats import get_rolling


def test_rolling_beta():
    returns = pd.Series(
        np.random.rand(10), index=pd.date_range("2020-01-01", periods=10)
    )
    underlying = pd.Series(
        np.random.rand(10), index=pd.date_range("2020-01-01", periods=10)
    )
    beta_series = get_rolling(
        returns, underlying, window=2, step=2, mode="beta", annualization_period=None
    )

    assert beta_series.shape[0] == 4


def test_rolling_alpha():
    returns = pd.Series(
        np.random.rand(10), index=pd.date_range("2020-01-01", periods=10)
    )
    underlying = pd.Series(
        np.random.rand(10), index=pd.date_range("2020-01-01", periods=10)
    )
    beta_series = get_rolling(
        returns,
        underlying,
        window=2,
        mode="default",
        step=2,
        annualization_period=252,
    )

    assert beta_series.shape[0] == 4
