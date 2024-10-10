from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


class SingleDecisionTree(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(
        self,
        threshold_margin: float,
        threshold_step: float,
        ensemble_num_trees: int | None,
        ensemble_percentile_gap: float | None,
        quantile_based: bool = True,
        aggregate_func: Literal["sum", "sharpe"] = "sharpe",
    ):
        assert threshold_margin < 0.5, f"Margin too large: {threshold_margin}"
        assert threshold_step < 0.2, f"Step too large: {threshold_margin}"

        self.threshold_to_test = np.arange(
            threshold_margin, 1 - threshold_margin, threshold_step
        ).tolist()
        self.quantile_based = quantile_based
        self.ensemble_num_trees = ensemble_num_trees
        self.ensemble_percentile_gap = ensemble_percentile_gap
        self.aggregate_func = aggregate_func
        if self.ensemble_num_trees is not None:
            assert self.ensemble_percentile_gap is not None, "Percentile gap required"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        assert X.shape[1] == 1, "Only single feature supported"
        X = X.squeeze()
        splits = (
            np.quantile(X, self.threshold_to_test, axis=0, method="closest_observation")
            if self.quantile_based
            else (self.threshold_to_test * (y.max() - y.min()) + y.min())
        )
        splits = np.unique(splits)
        if len(splits) == 1:
            self._best_split = splits[0]
            self._all_splits = [splits[0]]
            self._positive_class = 1
            return
        if len(splits) == 2:
            self._best_split = splits[0] - ((splits[1] - splits[0]) / 2)
            self._all_splits = [splits[0], splits[1]]
            self._positive_class = np.argmax(
                [np.mean(y[self._best_split > X]), np.mean(y[self._best_split <= X])]
            )
            return

        differences = [
            calculate_bin_diff(t, X=X, y=y, agg_method=self.aggregate_func)
            for t in splits
        ]
        self._best_split = float(splits[np.argmax(np.abs(differences))])
        self._all_splits = (
            _generate_neighbouring_splits(
                threshold=self.threshold_to_test[np.argmax(np.abs(differences))],
                num_trees=self.ensemble_num_trees,
                percentile_gap=self.ensemble_percentile_gap,  # type: ignore
                X=X,
            )
            if self.ensemble_num_trees is not None
            else [self._best_split]
        )
        self._positive_class = int(
            np.argmax(
                [np.mean(y[self._best_split > X]), np.mean(y[self._best_split <= X])]
            )
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._best_split is not None, "Model not fitted"
        assert self._positive_class is not None, "Model not fitted"
        assert self._all_splits is not None, "Model not fitted"
        other_class = 1 - self._positive_class
        return np.array(
            [
                np.where(X.squeeze() > split, self._positive_class, other_class)
                for split in self._all_splits
            ]
        ).mean(axis=0)


def _generate_neighbouring_splits(
    threshold: float, num_trees: int, percentile_gap: float, X: np.ndarray
) -> list[float]:
    thresholds = [threshold - percentile_gap, threshold, threshold + percentile_gap]
    if num_trees == 5:
        thresholds = [
            threshold - 2 * percentile_gap,
            threshold - percentile_gap,
            threshold,
            threshold + percentile_gap,
            threshold + 2 * percentile_gap,
        ]
    if num_trees == 7:
        thresholds = [
            threshold - 3 * percentile_gap,
            threshold - 2 * percentile_gap,
            threshold - percentile_gap,
            threshold,
            threshold + percentile_gap,
            threshold + 2 * percentile_gap,
            threshold + 3 * percentile_gap,
        ]
    if num_trees == 9:
        thresholds = [
            threshold - 4 * percentile_gap,
            threshold - 3 * percentile_gap,
            threshold - 2 * percentile_gap,
            threshold - percentile_gap,
            threshold,
            threshold + percentile_gap,
            threshold + 2 * percentile_gap,
            threshold + 3 * percentile_gap,
            threshold + 4 * percentile_gap,
        ]
    return [
        float(np.quantile(X, threshold, axis=0, method="closest_observation"))
        for threshold in thresholds
    ]


class RegularizedDecisionTree(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(
        self,
        threshold_margin: float,
        threshold_step: float,
        num_splits: int = 4,
        aggregate_func: Literal["mean", "sharpe"] = "sharpe",
    ):
        self.aggregate_func = aggregate_func
        assert threshold_margin <= 0.15, f"Margin too large: {threshold_margin}"
        assert threshold_step <= 0.05, f"Step too large: {threshold_margin}"
        self.num_splits = num_splits
        if threshold_margin > 0:
            threshold_margin = 0.5 - threshold_margin

            self.threshold_to_test = (
                np.arange(
                    threshold_margin, 1 - threshold_margin + 0.0001, threshold_step
                )
                .round(3)
                .tolist()
            )
        else:
            self.threshold_to_test = [0.5]

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        assert X.shape[1] == 1, "Only single feature supported"
        X = X.squeeze()
        splits = np.quantile(
            X, self.threshold_to_test, axis=0, method="closest_observation"
        )

        if isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        differences = [
            calculate_bin_diff(t, X=X, y=y, agg_method=self.aggregate_func)
            for t in splits
        ]
        idx_best_split = np.argmax(np.abs(differences))
        best_split = float(splits[idx_best_split])
        if np.isnan(best_split):
            self._splits = [splits[1]]
            self._positive_class = 1
            return

        self._positive_class = int(
            np.argmax(
                [
                    y[best_split > X].sum(),
                    y[best_split <= X].sum(),
                ]
            )
        )
        best_quantile = self.threshold_to_test[idx_best_split]
        deciles_to_split = (
            list(
                reversed(
                    [
                        best_quantile - (i * 0.01)
                        for i in range(0, 6 * self.num_splits, 5)
                    ][1:]
                )
            )
            + [best_quantile]
            + [best_quantile + (i * 0.01) for i in range(0, 6 * self.num_splits, 5)][1:]
        )
        self._splits = np.quantile(
            X,
            [round(i, 2) for i in deciles_to_split],
            axis=0,
            method="nearest",
        )
        assert np.isnan(self._splits).sum() == 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._positive_class is not None, "Model not fitted"
        assert self._splits is not None, "Model not fitted"

        output = np.searchsorted(self._splits, X.squeeze(), side="right") / len(
            self._splits
        )
        if isinstance(X, pd.DataFrame):
            output = pd.Series(output, index=X.index)
        if self._positive_class == 0:
            output = 1 - output
        return output


class UltraRegularizedDecisionTree(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(
        self,
        threshold_margin: float,  # used to produce the range of deciles/percentiles when the model can split, 0.1 means the range is 0.4 to 0.6 percentile.
        threshold_step: float,  # used to produce the range of deciles/percentiles when the model can split, 0.05 means the possible splits will be spaced 5% apart
        positive_class: int,  # this model can not flip the "coefficient", so the positive class is fixed
        num_splits: int = 4,  # number of extra splits to make around the best split, eg. if 2 and the best quantile is 0.5, then the splits will be [0.45, 0.5, 0.55]
        aggregate_func: Literal["mean", "sharpe"] = "sharpe",
    ):
        self.aggregate_func = aggregate_func
        assert threshold_margin <= 0.3, f"Margin too large: {threshold_margin}"
        assert threshold_step <= 0.05, f"Step too large: {threshold_margin}"
        self.num_splits = num_splits
        self.positive_class = positive_class
        if threshold_margin > 0:
            threshold_margin = 0.5 - threshold_margin

            self.threshold_to_test = (
                np.arange(
                    threshold_margin, 1 - threshold_margin + 0.0001, threshold_step
                )
                .round(3)
                .tolist()
            )
        else:
            self.threshold_to_test = [0.5]

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        assert X.shape[1] == 1, "Only single feature supported"
        X = X.squeeze()
        splits = np.quantile(
            X, self.threshold_to_test, axis=0, method="closest_observation"
        )
        differences = [
            calculate_bin_diff(t, X=X, y=y, agg_method=self.aggregate_func)
            for t in splits
        ]
        idx_best_split = np.argmax(np.abs(differences))
        best_split = float(splits[idx_best_split])
        if np.isnan(best_split):
            self._splits = [splits[1]]
            return

        best_quantile = self.threshold_to_test[idx_best_split]
        deciles_to_split = (
            list(
                reversed(
                    [
                        best_quantile - (i * 0.01)
                        for i in range(0, 6 * self.num_splits, 5)
                    ][1:]
                )
            )
            + [best_quantile]
            + [best_quantile + (i * 0.01) for i in range(0, 6 * self.num_splits, 5)][1:]
        )  # number of extra splits to make around the best split, eg. if 2 and the best quantile is 0.5, then the splits will be [0.45, 0.5, 0.55]
        self._splits = np.quantile(
            X,
            [round(i, 2) for i in deciles_to_split],
            axis=0,
            method="nearest",
        )  # translate the percentiles into actual values
        assert np.isnan(self._splits).sum() == 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert self.positive_class is not None, "Model not fitted"
        assert self._splits is not None, "Model not fitted"

        output = (
            np.searchsorted(self._splits, X.squeeze(), side="right") / len(self._splits)
        )  # find the value in the splits, the index of the split acts as a scaled value between 0 and 1
        if isinstance(X, pd.DataFrame):
            output = pd.Series(output, index=X.index)
        if self.positive_class == 0:
            output = 1 - output
        return output


class TwoDimensionalPiecewiseLinearRegression(
    BaseEstimator, ClassifierMixin, MultiOutputMixin
):
    def __init__(
        self,
        # used to produce the range of deciles/percentiles when the model can split, 0.1 means the range is 0.4 to 0.6 percentile.
        exogenous_threshold_margin: float,
        endogenous_threshold_margin: float,
        # used to produce the range of deciles/percentiles when the model can split, 0.05 means the possible splits will be spaced 5% apart
        exogenous_threshold_step: float,
        endogenous_threshold_step: float,
        # this model can not flip the "coefficient", so the positive class is fixed
        exogenous_positive_class: int,
        endogenous_positive_class: int,
        # number of extra splits to make around the best split, eg. if 2 and the best quantile is 0.5, then the splits will be [0.45, 0.5, 0.55]
        exogenous_num_splits: int = 4,
        endogenous_num_splits: int = 4,
        aggregate_func: Literal["mean", "sharpe"] = "mean",
    ):
        self.aggregate_func = aggregate_func
        assert (
            exogenous_threshold_margin <= 0.3
        ), f"{exogenous_threshold_margin=} too large (> 0.3)"
        assert (
            endogenous_threshold_margin <= 0.3
        ), f"{endogenous_threshold_margin=} too large (> 0.3)"
        assert (
            0 < exogenous_threshold_step <= 0.05
        ), f"{exogenous_threshold_step=} too large (> 0.05) or negative"
        assert (
            0 < endogenous_threshold_step <= 0.05
        ), f"{endogenous_threshold_step=} too large (> 0.05) or negative"
        self._exogenous_positive_class = exogenous_positive_class
        self._endogenous_positive_class = endogenous_positive_class
        self.exogenous_num_splits = exogenous_num_splits
        self.endogenous_num_splits = endogenous_num_splits

        if exogenous_threshold_margin > 0:
            exogenous_threshold_margin = 0.5 - exogenous_threshold_margin

            self.exogenous_thresholds_to_test = (
                np.arange(
                    exogenous_threshold_margin,
                    1 - exogenous_threshold_margin + 0.0001,
                    exogenous_threshold_step,
                )
                .round(3)
                .tolist()
            )
        else:
            self.exogenous_thresholds_to_test = [0.5]

        if endogenous_threshold_margin > 0:
            endogenous_threshold_margin = 0.5 - endogenous_threshold_margin

            self.endogenous_thresholds_to_test = (
                np.arange(
                    endogenous_threshold_margin,
                    1 - endogenous_threshold_margin + 0.0001,
                    endogenous_threshold_step,
                )
                .round(3)
                .tolist()
            )
        else:
            self.endogenous_thresholds_to_test = [0.5]

        self._exogenous_splits = None
        self._endogenous_splits = None

    def fit(  # noqa: PLR0912
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ):
        assert X.shape[1] == 2, "Exactly two features are supported"
        self._exogenous_X_col = 0
        self._endogenous_X_col = 1

        assert (
            X[:, self._exogenous_X_col].var() != 0
        ), f"{self._exogenous_X_col=} has no variance"
        assert (
            X[:, self._endogenous_X_col].var() != 0
        ), f"{self._endogenous_X_col=} has no variance"

        exogenous_splits = np.quantile(
            X[:, self._exogenous_X_col],
            self.exogenous_thresholds_to_test,
            axis=0,
            method="closest_observation",
        )
        endogenous_splits = np.quantile(
            X[:, self._endogenous_X_col],
            self.endogenous_thresholds_to_test,
            axis=0,
            method="closest_observation",
        )

        exogenous_best_split_idx = None
        endogenous_best_split_idx = None
        highest_abs_difference = None

        for exogenous_split_idx, exogenous_split in enumerate(exogenous_splits):
            # It could be that the best split comes from considering only the first column in X, not both.
            exogenous_difference = calculate_bin_diff(
                exogenous_split,
                X=X[:, self._exogenous_X_col],
                y=y,
                agg_method=self.aggregate_func,
            )

            if (
                highest_abs_difference is None
                or abs(exogenous_difference) > highest_abs_difference
            ):
                highest_abs_difference = abs(exogenous_difference)
                exogenous_best_split_idx = exogenous_split_idx
                endogenous_best_split_idx = None

            # It could be that the best split comes from considering both columns in X.
            for endogenous_split_idx, endogenous_split in enumerate(endogenous_splits):
                differences = calculate_2d_bin_diff(
                    quantile_exogenous=exogenous_split,
                    quantile_endogenous=endogenous_split,
                    X=X,
                    y=y,
                    agg_method=self.aggregate_func,
                )
                if (
                    highest_abs_difference is None
                    or abs(differences) > highest_abs_difference
                ):
                    highest_abs_difference = abs(differences)
                    exogenous_best_split_idx = exogenous_split_idx
                    endogenous_best_split_idx = endogenous_split_idx

        if exogenous_best_split_idx is None and endogenous_best_split_idx is None:
            self._exogenous_splits = [exogenous_splits[0]]
            self._endogenous_splits = [endogenous_splits[0]]
            return

        exogenous_deciles_to_split = None
        if exogenous_best_split_idx is not None:
            exogenous_deciles_to_split = calc_deciles_to_split(
                best_quantile=self.exogenous_thresholds_to_test[
                    exogenous_best_split_idx
                ],
                num_splits=self.exogenous_num_splits,
            )

        endogenous_deciles_to_split = None
        if endogenous_best_split_idx is not None:
            endogenous_deciles_to_split = calc_deciles_to_split(
                best_quantile=self.endogenous_thresholds_to_test[
                    endogenous_best_split_idx
                ],
                num_splits=self.endogenous_num_splits,
            )

        if exogenous_best_split_idx is None:
            # raise ValueError("exogenous_best_split_idx is None")
            self._exogenous_splits = None
        else:
            self._exogenous_splits = np.quantile(
                X[:, self._exogenous_X_col],
                exogenous_deciles_to_split,
                axis=0,
                method="nearest",
            )  # translate the percentiles into actual values
            assert np.isnan(self._exogenous_splits).sum() == 0

        if endogenous_best_split_idx is None:
            # raise ValueError("endogenous_best_split_idx is None")
            self._endogenous_splits = None
        else:
            self._endogenous_splits = np.quantile(
                X[:, self._endogenous_X_col],
                endogenous_deciles_to_split,
                axis=0,
                method="nearest",
            )  # translate the percentiles into actual values
            assert np.isnan(self._endogenous_splits).sum() == 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert X.shape[1] == 2, "Exactly two features are supported"
        assert self._exogenous_positive_class is not None, "Model not fitted"
        assert self._endogenous_positive_class is not None, "Model not fitted"
        assert (
            self._exogenous_splits is not None or self._endogenous_splits is not None
        ), "Model not fitted"

        if self._exogenous_splits is None:
            exogenous_output = None
        else:
            exogenous_output = np.searchsorted(
                self._exogenous_splits,
                X[:, self._exogenous_X_col],
                side="right",
            ) / len(self._exogenous_splits)
            if self._exogenous_positive_class == 0:
                exogenous_output = 1 - exogenous_output

        if self._endogenous_splits is None:
            endogenous_output = None
        else:
            endogenous_output = np.searchsorted(
                self._endogenous_splits,
                X[:, self._endogenous_X_col],
                side="right",
            ) / len(self._endogenous_splits)
            if self._endogenous_positive_class == 0:
                endogenous_output = 1 - endogenous_output

        if exogenous_output is not None and endogenous_output is not None:
            output = (exogenous_output + endogenous_output) / 2
        elif exogenous_output is not None:
            output = exogenous_output
        else:  # endogenous_output is not None
            output = endogenous_output

        return output


def calculate_bin_diff(
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
