from typing import Literal

import numpy as np
import pandas as pd
from finml_utils.models.utils import (
    calc_deciles_to_split,
    calculate_1d_bin_diff,
    calculate_2d_bin_diff,
)
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


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

    def fit(
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
            exogenous_difference = calculate_1d_bin_diff(
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
