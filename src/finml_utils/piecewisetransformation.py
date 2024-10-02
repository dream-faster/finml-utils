import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin


class PiecewiseLinearTransformation(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(
        self,
        positive_class: int,
        num_splits: int = 4,
    ):
        self.num_splits = num_splits
        self.positive_class = positive_class

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ):
        assert X.shape[1] == 1, "Only single feature supported"
        X = X.squeeze()

        if isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        deciles_to_split = (
            list(
                reversed(
                    [0.5 - (i * 0.01) for i in range(0, 6 * self.num_splits, 5)][1:]
                )
            )
            + [0.5]
            + [0.5 + (i * 0.01) for i in range(0, 6 * self.num_splits, 5)][1:]
        )
        self._splits = np.quantile(
            X,
            [round(i, 2) for i in deciles_to_split],
            axis=0,
            method="nearest",
        )
        assert np.isnan(self._splits).sum() == 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert self._splits is not None, "Model not fitted"
        output = np.searchsorted(self._splits, X.squeeze(), side="right") / len(
            self._splits
        )
        output = pd.Series(output, index=X.index)
        if self.positive_class == 0:
            output = 1 - output
        return output
