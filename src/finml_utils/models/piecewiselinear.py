from typing import Literal

import numpy as np
import pandas as pd
from finml_utils.models.utils import calculate_1d_bin_diff
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


class PiecewiseLinearRegression(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(
        self,
        threshold_margin: float,
        threshold_step: float,
        num_splits: int = 4,
        aggregate_func: Literal["mean", "sharpe"] = "sharpe",
    ):
        self.aggregate_func = aggregate_func
        self.threshold_margin = threshold_margin
        self.threshold_step = threshold_step
        assert threshold_margin <= 0.3, f"Margin too large: {threshold_margin}"
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
            calculate_1d_bin_diff(t, X=X, y=y, agg_method=self.aggregate_func)
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


RegularizedDecisionTree = PiecewiseLinearRegression


class PiecewiseLinearRegressionMonotonic(
    BaseEstimator, ClassifierMixin, MultiOutputMixin
):
    def __init__(
        self,
        threshold_margin: float,  # used to produce the range of deciles/percentiles when the model can split, 0.1 means the range is 0.4 to 0.6 percentile.
        threshold_step: float,  # used to produce the range of deciles/percentiles when the model can split, 0.05 means the possible splits will be spaced 5% apart
        positive_class: int,  # this model can not flip the "coefficient", so the positive class is fixed
        num_splits: int = 4,  # number of extra splits to make around the best split, eg. if 2 and the best quantile is 0.5, then the splits will be [0.45, 0.5, 0.55]
        aggregate_func: Literal["mean", "sharpe"] = "sharpe",
    ):
        self.aggregate_func = aggregate_func
        self.threshold_margin = threshold_margin
        self.threshold_step = threshold_step
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
            calculate_1d_bin_diff(t, X=X, y=y, agg_method=self.aggregate_func)
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


UltraRegularizedDecisionTree = PiecewiseLinearRegressionMonotonic
