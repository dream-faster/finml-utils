from math import isnan, sqrt

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, rf=0.0, periods=252, annualize=True):
    if rf != 0 and periods is None:
        raise NotImplementedError("Must provide periods if rf != 0")

    divisor = returns.std(ddof=1)
    res = returns.mean() / divisor

    if isnan(res):
        return -10.0

    if annualize:
        return res * sqrt(periods * get_avg_timestamps_per_day(returns.index))

    return res


def sortino(returns, periods=252, annualize=True):
    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    res = returns.mean() / downside

    if annualize:
        return res * sqrt(periods * get_avg_timestamps_per_day(returns.index))

    return res


def get_avg_timestamps_per_day(index: pd.DatetimeIndex) -> float:
    return len(index) / len(np.unique(index.date))


def information_ratio(returns, benchmark):
    diff_rets = returns - benchmark
    return diff_rets.mean() / diff_rets.std()


def is_outlier(points: pd.DataFrame, thresh: int) -> pd.Series:
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    """
    median = points.median()
    diff = (points - median).abs()
    med_deviation = diff.median()

    modified_z_score = 0.6745 * diff / med_deviation

    return modified_z_score > thresh


def get_number_of_observations(df: pd.DataFrame) -> pd.Series:
    return (df.diff().abs() > 0).sum()


def get_frequency_of_change(df: pd.DataFrame) -> pd.Series:
    return get_number_of_observations(df) / df.notna().sum()


def compsum(returns):
    return returns.add(1).cumprod() - 1


def to_prices(returns: pd.Series, base=1.0) -> pd.Series:
    returns = returns.copy().fillna(0).replace([np.inf, -np.inf], float("NaN"))
    return base + base * compsum(returns)