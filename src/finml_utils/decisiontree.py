import math
from typing import Literal, Tuple

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
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ):
        assert X.shape[1] == 1, "Only single feature supported"
        X = X.squeeze()
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert self._best_split is not None, "Model not fitted"
        assert self._positive_class is not None, "Model not fitted"
        assert self._all_splits is not None, "Model not fitted"
        other_class = 1 - self._positive_class
        return pd.Series(
            np.array(
                [
                    np.where(X.squeeze() > split, self._positive_class, other_class)
                    for split in self._all_splits
                ]
            ).mean(axis=0),
            index=X.index,
        )


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
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
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
        self._positive_class = positive_class
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
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
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
        assert self._positive_class is not None, "Model not fitted"
        assert self._splits is not None, "Model not fitted"

        output = (
            np.searchsorted(self._splits, X.squeeze(), side="right") / len(self._splits)
        )  # find the value in the splits, the index of the split acts as a scaled value between 0 and 1
        if isinstance(X, pd.DataFrame):
            output = pd.Series(output, index=X.index)
        if self._positive_class == 0:
            output = 1 - output
        return output


class TwoDimPiecewiseDecisionTree(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(
        self,
        threshold_margins: Tuple[float, float],  # used to produce the range of deciles/percentiles when the model can split, 0.1 means the range is 0.4 to 0.6 percentile.
        threshold_steps: Tuple[float, float],  # used to produce the range of deciles/percentiles when the model can split, 0.05 means the possible splits will be spaced 5% apart
        positive_classes: Tuple[int, int],  # this model can not flip the "coefficient", so the positive class is fixed
        num_splitss: Tuple[int, int] = (4, 4),  # number of extra splits to make around the best split, eg. if 2 and the best quantile is 0.5, then the splits will be [0.45, 0.5, 0.55]
        aggregate_func: Literal["mean", "sharpe"] = "sharpe",
    ):
        self.aggregate_func = aggregate_func
        assert len(threshold_margins) == 2, f"{len(threshold_margins)=} instead of 2"
        assert len(threshold_steps) == 2, f"{len(threshold_steps)=} instead of 2"
        assert len(positive_classes) == 2, f"{len(positive_classes)=} instead of 2"
        assert len(num_splitss) == 2, f"{len(num_splitss)=} instead of 2"
        assert all(
            threshold_margin <= 0.3 for threshold_margin in threshold_margins
        ), f"Margin(s) too large: {threshold_margins}"
        assert all(
            threshold_step <= 0.05 for threshold_step in threshold_steps
        ), f"Step(s) too large: {threshold_margins}"
        self.num_splitss = num_splitss
        self._positive_classes = positive_classes

        self.thresholds_to_test = []
        for threshold_margin, threshold_step in zip(threshold_margins, threshold_steps):
            if threshold_margin > 0:
                threshold_margin = 0.5 - threshold_margin

                self.thresholds_to_test.append(
                    np.arange(
                        threshold_margin, 1 - threshold_margin + 0.0001, threshold_step
                    )
                    .round(3)
                    .tolist()
                )
            else:
                self.thresholds_to_test.append([0.5])
        self.thresholds_to_test = tuple(self.thresholds_to_test)

        self._splits_0 = None
        self._splits_1 = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ):
        assert X.shape[1] == 2, "Exactly two features are supported"
        X = X.squeeze()
        splitss = [
            np.quantile(
                X[col], threshold_to_test, axis=0, method="closest_observation"
            )
            for col, threshold_to_test
            in zip(X.columns, self.thresholds_to_test)
        ]
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        idx_best_split_0 = None
        idx_best_split_1 = None
        highest_abs_difference = None
        # It could be that the best split comes from considering only the second column in X, not both.
        for idx_1, split_1 in enumerate(splitss[1]):
            difference_1 = calculate_bin_diff(
                quantile=split_1, X=X[X.columns[1]], y=y, agg_method=self.aggregate_func
            )
            if highest_abs_difference is None or abs(difference_1) > highest_abs_difference:
                highest_abs_difference = abs(difference_1)
                idx_best_split_0 = None
                idx_best_split_1 = idx_1

        for idx_0, split_0 in enumerate(splitss[0]):
            # It could be that the best split comes from considering only the first column in X, not both.
            difference_0 = calculate_bin_diff(
                quantile=split_0, X=X[X.columns[0]], y=y, agg_method=self.aggregate_func
            )
            if highest_abs_difference is None or abs(difference_0) > highest_abs_difference:
                highest_abs_difference = abs(difference_0)
                idx_best_split_0 = idx_0
                idx_best_split_1 = None

            # It could be that the best split comes from considering both columns in X.
            for idx_1, split_1 in enumerate(splitss[1]):
                difference = calculate_bin_diff(
                    quantile_0=split_0, quantile_1=split_1, X=X, y=y, agg_method=self.aggregate_func
                )
                if highest_abs_difference is None or abs(difference) > highest_abs_difference:
                    highest_abs_difference = abs(difference)
                    idx_best_split_0 = idx_0
                    idx_best_split_1 = idx_1

        if idx_best_split_0 is None and idx_best_split_1 is None:
            self._splits_0 = [splitss[0][1]]
            self._splits_1 = [splitss[1][1]]
            return

        pos_times = 6

        deciles_to_split_0 = None
        if idx_best_split_0 is not None:
            best_quantile_0 = self.thresholds_to_test[0][idx_best_split_0]
            deciles_to_split_0 = calc_deciles_to_split(best_quantile_0, self.num_splitss[0])

        deciles_to_split_1 = None
        if idx_best_split_1 is not None:
            best_quantile_1 = self.thresholds_to_test[1][idx_best_split_1]
            deciles_to_split_1 = calc_deciles_to_split(best_quantile_1, self.num_splitss[1])

        # In case best split is created using values from columns,
        # then reduce the number of deciles around the best split.
        if idx_best_split_0 is not None and idx_best_split_1 is not None:
            n_pos_deciles_range_0 = int(math.sqrt(len(deciles_to_split_0)) / 2)
            deciles_range_0_mid_idx = len(deciles_to_split_0) // 2
            deciles_to_split_0 = deciles_to_split_0[
                deciles_range_0_mid_idx - n_pos_deciles_range_0
                : deciles_range_0_mid_idx + n_pos_deciles_range_0 + 1
            ]

            n_pos_deciles_range_1 = int(math.sqrt(len(deciles_to_split_1)) / 2)
            deciles_range_1_mid_idx = len(deciles_to_split_1) // 2
            deciles_to_split_1 = deciles_to_split_1[
                deciles_range_1_mid_idx - n_pos_deciles_range_1
                : deciles_range_1_mid_idx + n_pos_deciles_range_1 + 1
            ]

        if idx_best_split_0 is None:
            self._splits_0 = None
        else:
            self._splits_0 = np.quantile(
                X[X.columns[0]],
                [decile_to_split_0 for decile_to_split_0 in deciles_to_split_0],
                axis=0,
                method="nearest",
            )  # translate the percentiles into actual values
            assert np.isnan(self._splits_0).sum() == 0

        if idx_best_split_1 is None:
            self._splits_1 = None
        else:
            self._splits_1 = np.quantile(
                X[X.columns[1]],
                [decile_to_split_1 for decile_to_split_1 in deciles_to_split_1],
                axis=0,
                method="nearest",
            )  # translate the percentiles into actual values
            assert np.isnan(self._splits_1).sum() == 0

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO ask: batch predict won't work here.
        assert X.shape[1] == 2, "Exactly two features are supported"
        assert self._positive_classes is not None, "Model not fitted"
        assert (self._splits_0 is not None or self._splits_1 is not None), "Model not fitted"

        if self._splits_0 is None:
            output_col_0 = np.repeat(np.nan, X.shape[0])
        else:
            output_col_0 = \
                np.searchsorted(self._splits_0, X[X.columns[0]].squeeze(), side="right") / len(self._splits_0)
            if self._positive_classes[0] == 0:
                output_col_0 = 1 - output_col_0

        if self._splits_1 is None:
            output_col_1 = np.repeat(np.nan, X.shape[0])
        else:
            output_col_1 = \
                np.searchsorted(self._splits_1, X[X.columns[1]].squeeze(), side="right") / len(self._splits_1)
            if self._positive_classes[1] == 0:
                output_col_1 = 1 - output_col_1

        return pd.DataFrame(
            data={
                X.columns[0]: output_col_0,
                X.columns[1]: output_col_1,
            },
            index=X.index
        )


def calculate_bin_diff(
    quantile: float,
    X: np.ndarray,
    y: np.ndarray,
    agg_method: Literal["mean", "sharpe"],
) -> float:
    above = quantile > X

    agg = np.array([np_mean(y[~above]), np_mean(y[above])])
    if agg_method == "sharpe":
        agg = agg / np.array([np_std(y[~above]), np_std(y[above])])
    if len(agg) == 0:
        return 0.0
    if len(agg) == 1:
        return 0.0
    if len(agg) > 2:
        raise AssertionError("Too many bins")
    return np.diff(agg)[0]


def calculate_bin_diff(
    quantile_0: float,
    quantile_1: float,
    X: pd.DataFrame,
    y: np.ndarray,
    agg_method: Literal["mean", "sharpe"],
) -> float:
    above = (quantile_0 > X[X.columns[0]]) & (quantile_1 > X[X.columns[1]])

    agg = np.array([np_mean(y[~above]), np_mean(y[above])])
    if agg_method == "sharpe":
        agg = agg / np.array([np_std(y[~above]), np_std(y[above])])
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
    range_start = pos_times * num_splits

    neg_times = -(range_start // range_step)
    if range_start % range_step == 0:
        neg_times += 1
    range_stop = neg_times * range_step

    return [round(best_quantile + (i * 0.01), 2) for i in range(range_stop, range_start, range_step)]