import numpy as np
import pandas as pd
from finml_utils.decisiontree import (
    RegularizedDecisionTree,
    SingleDecisionTree,
    UltraRegularizedDecisionTree,
)
from finml_utils.piecewisetransformation import PiecewiseLinearTransformation


def test_singledecisiontree():
    model = SingleDecisionTree(
        threshold_margin=0.3,
        threshold_step=0.05,
        ensemble_num_trees=None,
        ensemble_percentile_gap=None,
    )
    model.fit(
        X=pd.DataFrame(np.array([[3.0, 2.0, 0.4, 0.6, -1.0, 0.0, 0.2, 1.2, 1.0]]).T),
        y=pd.Series([1, 1, 1, 0, 0, 0, 0, 1, 1]),
        sample_weight=None,
    )
    print(model._best_split)


def test_regularizeddecisiontree():
    model = RegularizedDecisionTree(
        threshold_margin=0.1, threshold_step=0.02, num_splits=4
    )
    assert model.threshold_to_test == [
        0.4,
        0.42,
        0.44,
        0.46,
        0.48,
        0.5,
        0.52,
        0.54,
        0.56,
        0.58,
        0.6,
    ]
    X = pd.DataFrame(np.arange(-9, 10, 1).T)
    y = pd.Series((np.arange(-9, 10, 1).T) * 0.1)
    model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    preds = model.predict(X)

    inverse_model = RegularizedDecisionTree(threshold_margin=0.1, threshold_step=0.05)
    inverse_model.fit(
        X=X,
        y=1 - y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)


def test_ultraregularizeddecisiontree():
    model = UltraRegularizedDecisionTree(
        threshold_margin=0.1, threshold_step=0.05, num_splits=4, positive_class=1
    )
    # assert model.threshold_to_test == [
    #     0.5,
    # ]
    X = pd.DataFrame(np.arange(-9, 10, 1).T)
    y = pd.Series((np.arange(-9, 10, 1).T) * 0.1)
    model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    preds = model.predict(X)

    inverse_model = UltraRegularizedDecisionTree(
        threshold_margin=0.0, threshold_step=0.02, num_splits=4, positive_class=0
    )
    inverse_model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)


def test_piecewisetransformation():
    model = PiecewiseLinearTransformation(num_splits=8, positive_class=1)

    X = pd.DataFrame(np.arange(-9, 10, 1).T)
    y = pd.Series((np.arange(-9, 10, 1).T) * 0.1)
    model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    preds = model.predict(X)

    inverse_model = PiecewiseLinearTransformation(num_splits=8, positive_class=0)
    inverse_model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)
