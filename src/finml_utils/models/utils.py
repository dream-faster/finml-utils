from typing import Literal

import numpy as np


def calculate_1d_bin_diff(
    quantile: float,
    X: np.ndarray,
    y: np.ndarray,
    agg_method: Literal["mean", "sharpe"],
) -> float:
    above = quantile > X
    return _calculate_bin_diff(above, y, agg_method)


def calculate_2d_bin_diff(
    quantile_exogenous: float,
    quantile_endogenous: float,
    X: np.ndarray,
    y: np.ndarray,
    agg_method: Literal["mean", "sharpe"],
) -> float:
    above = (quantile_exogenous > X[:, 0]) & (quantile_endogenous > X[:, 1])
    return _calculate_bin_diff(above, y, agg_method)


def _calculate_bin_diff(
    above: np.ndarray,
    y: np.ndarray,
    agg_method: Literal["mean", "sharpe"],
):
    y_below = y[~above]
    y_above = y[above]

    # Calling code ensures that len(y_below) != 0 and len(y_above) != 0.
    agg = np.array([np_mean(y_below), np_mean(y_above)])

    if agg_method == "sharpe":
        std = np.array([np_std(y[~above]), np_std(y[above])])
        agg = agg / std

    if len(agg) == 0:
        return 0.0
    if len(agg) == 1:
        return 0.0
    if len(agg) > 2:
        raise AssertionError("Too many bins")

    return np.diff(agg)[0]


def np_mean(x):
    if x.size == 0:
        return 0.0
    return x.mean()


def np_std(x):
    if x.size == 0:
        return 0.0
    return x.std()


def calc_deciles_to_split(best_quantile: float, num_splits: int) -> list[float]:
    """
    Example:
    - If best_quantile = 0.45 and num_splits = 3, then result is [0.30, 0.35, 0.40] + [0.45] + [0.50, 0.55, 0.60]
    - If best_quantile = 0.51 and num_splits = 2, then result is [0.41, 0.46] + [0.51] + [0.56, 0.61]

    :param best_quantile: The best quantile to add splits around
    :param num_splits: The number of splits around `best_quantile` to add in 0.05 steps
    :return: An array sorted in ascending manner, containing `num_splits * 2 + 1` elements:
        - `num_splits` values 0.05 apart before `best_quantile`, and
        - `best_quantile`, and
        - `num_splits` values 0.05 apart after `best_quantile`
    """

    range_step = 5

    pos_times = 6
    range_stop = pos_times * num_splits

    neg_times = -(range_stop // range_step)
    if range_stop % range_step == 0:
        neg_times += 1
    range_start = neg_times * range_step

    return [
        round(best_quantile + (i * 0.01), 2)
        for i in range(range_start, range_stop, range_step)
    ]
