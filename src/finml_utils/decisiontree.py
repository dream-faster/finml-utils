from typing import Literal

import numpy as np
import numpy_groupies as npg
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

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None):
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

        def calculate_bin_diff(
            quantile: float,
            X: np.ndarray,
            y: np.ndarray,
            agg_method: Literal["sum", "sharpe"],
        ):
            signal = np.where(quantile > X, 1, 0)
            agg = npg.aggregate(signal, y, func="sum")
            if agg_method == "sharpe":
                agg = agg / npg.aggregate(signal, y, func="std")
            if len(agg) == 0:
                return 0.0
            if len(agg) == 1:
                return 0.0
            if len(agg) > 2:
                raise AssertionError("Too many bins")
            return np.diff(agg)[0]

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
        threshold = [
            threshold - 2 * percentile_gap,
            threshold - percentile_gap,
            threshold,
            threshold + percentile_gap,
            threshold + 2 * percentile_gap,
        ]
    return [
        float(np.quantile(X, threshold, axis=0, method="closest_observation"))
        for threshold in thresholds
    ]
