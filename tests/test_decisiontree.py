import numpy as np
import pandas as pd
from finml_utils.decisiontree import SingleDecisionTree


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
