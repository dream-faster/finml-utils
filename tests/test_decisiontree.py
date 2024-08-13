import numpy as np
import pandas as pd
from finml_utils.decisiontree import RegularizedDecisionTree, SingleDecisionTree


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


def test_diversifieddecisiontree():
    model = RegularizedDecisionTree(thresholds_to_test=[0.4, 0.5, 0.6])
    X = pd.DataFrame(
        np.array([[4.0, 2.0, 3.0, 0.4, 0.6, -1.0, -2.0, -3.0, 0.0, 0.2, 1.2, 1.0]]).T
    )
    y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    preds = model.predict(X)

    inverse_model = RegularizedDecisionTree(thresholds_to_test=[0.4, 0.5, 0.6])
    inverse_model.fit(
        X=X,
        y=1 - y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)
