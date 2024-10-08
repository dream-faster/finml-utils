import itertools
from typing import Literal

import numpy as np
import pandas as pd
from finml_utils.decisiontree import (
    RegularizedDecisionTree,
    SingleDecisionTree,
    UltraRegularizedDecisionTree,
)
from finml_utils.piecewisetransformation import PiecewiseLinearTransformation

import pytest
from hypothesis import given, strategies as st
from hypothesis import assume as hypothesis_assume

from src.finml_utils import TwoDimensionalPiecewiseLinearRegression


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
        threshold_margin=0.1, threshold_step=0.05, num_splits=4, positive_class=0
    )
    inverse_model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)


@st.composite
def same_len_X_y_lists(draw):
    list_len = draw(st.integers(min_value=10, max_value=100))
    fixed_len_X_col_list = st.lists(st.floats(min_value=-3, max_value=3), min_size=list_len, max_size=list_len)
    fixed_len_y_list = st.lists(st.booleans(), min_size=list_len, max_size=list_len)

    return (
        draw(fixed_len_X_col_list),  # exogenous_X_col_for_fit
        draw(fixed_len_X_col_list),  # endogenous_X_col_for_fit
        draw(fixed_len_y_list),      # bool_y
        draw(fixed_len_X_col_list),  # exogenous_X_col_for_predict
        draw(fixed_len_X_col_list)   # endogenous_X_col_for_predict
    )


@given(
    st.integers(min_value=-1, max_value=30),   # exogenous_threshold_margin_x100
    st.integers(min_value=-1, max_value=30),   # endogenous_threshold_margin_x100
    st.integers(min_value=1, max_value=5),      # exogenous_threshold_step_x100
    st.integers(min_value=1, max_value=5),      # endogenous_threshold_step_x100
    st.booleans(),                              # bool_exogenous_positive_class
    st.booleans(),                              # bool_endogenous_positive_class
    st.integers(min_value=1, max_value=5),      # exogenous_num_splits
    st.integers(min_value=1, max_value=5),      # endogenous_num_splits
    same_len_X_y_lists(),                       # same_len_X_y_lists
    st.booleans(),                              # is_aggregate_func_mean
)
def test_twodimensionalpiecewiselinearregression(
    exogenous_threshold_margin_x100: int,
    endogenous_threshold_margin_x100: int,
    exogenous_threshold_step_x100: int,
    endogenous_threshold_step_x100: int,
    bool_exogenous_positive_class: bool,
    bool_endogenous_positive_class: bool,
    exogenous_num_splits: int,
    endogenous_num_splits: int,
    same_len_X_y_lists: tuple[list[float], list[float], list[bool], list[float], list[float]],
    is_aggregate_func_mean: bool
):
    exogenous_threshold_margin = round(exogenous_threshold_margin_x100 / 100, 3)
    endogenous_threshold_margin = round(endogenous_threshold_margin_x100 / 100, 3)
    exogenous_threshold_step = round(exogenous_threshold_step_x100 / 100, 3)
    endogenous_threshold_step = round(endogenous_threshold_step_x100 / 100, 3)
    exogenous_positive_class = bool(bool_exogenous_positive_class)
    endogenous_positive_class = bool(bool_endogenous_positive_class)
    aggregate_func: Literal["mean", "sharpe"] = "mean" if is_aggregate_func_mean else "sharpe"

    def get_2d_model() -> TwoDimensionalPiecewiseLinearRegression:
        return TwoDimensionalPiecewiseLinearRegression(
            exogenous_threshold_margin,
            endogenous_threshold_margin,
            exogenous_threshold_step,
            endogenous_threshold_step,
            exogenous_positive_class,
            endogenous_positive_class,
            exogenous_num_splits,
            endogenous_num_splits,
            aggregate_func,
        )

    (
        exogenous_X_col_for_fit,
        endogenous_X_col_for_fit,
        bool_y,
        exogenous_X_col_for_predict,
        endogenous_X_col_for_predict
    ) = same_len_X_y_lists

    y = pd.Series([int(bool_y) for bool_y in bool_y])
    X_for_predict = pd.DataFrame(data={
        "exogenous": exogenous_X_col_for_predict,
        "endogenous": endogenous_X_col_for_predict,
    })

    d2_model = get_2d_model()

    if exogenous_threshold_margin <= 0:
        assert d2_model.exogenous_thresholds_to_test == [0.5]
    else:
        for i, threshold in enumerate(d2_model.exogenous_thresholds_to_test):
            assert threshold == round(0.5 - exogenous_threshold_margin + i * exogenous_threshold_step, 3)

    if endogenous_threshold_margin <= 0:
        assert d2_model.endogenous_thresholds_to_test == [0.5]
    else:
        for i, threshold in enumerate(d2_model.endogenous_thresholds_to_test):
            assert threshold == round(0.5 - endogenous_threshold_margin + i * endogenous_threshold_step, 3)

    # Test as if exogenous X column values were constants.

    endogenous_only_X_for_fit = pd.DataFrame(data={
        "exogenous": [0] * len(endogenous_X_col_for_fit),
        "endogenous": endogenous_X_col_for_fit,
    })
    d2_model.fit(X=endogenous_only_X_for_fit, y=y)

    assert d2_model._X_cols == [d2_model._exogenous_X_col, d2_model._endogenous_X_col]

    d1_model = UltraRegularizedDecisionTree(
        threshold_margin=endogenous_threshold_margin,
        threshold_step=endogenous_threshold_step,
        positive_class=endogenous_positive_class,
        num_splits=endogenous_positive_class,
        aggregate_func=aggregate_func,
    )
    d1_model.fit(X=endogenous_only_X_for_fit[["endogenous"]], y=y)

    assert d2_model.endogenous_thresholds_to_test == d1_model.threshold_to_test
    assert d2_model._exogenous_splits is None
    assert len(d2_model._endogenous_splits) == len(d1_model._splits)
    assert (d2_model._endogenous_splits == d1_model._splits).all()

    d2_y_pred = d2_model.predict(X_for_predict)
    d1_y_pred = d1_model.predict(X_for_predict[["endogenous"]])

    assert np.allclose(d2_y_pred, d1_y_pred)

    # Test as if endogenous X column values were constants.

    d2_model = get_2d_model()
    exogenous_only_X_for_fit = pd.DataFrame(data={
        "exogenous": exogenous_X_col_for_fit,
        "endogenous": [0] * len(exogenous_X_col_for_fit),
    })
    d2_model.fit(X=exogenous_only_X_for_fit, y=y)

    d1_model = UltraRegularizedDecisionTree(
        threshold_margin=exogenous_threshold_margin,
        threshold_step=exogenous_threshold_step,
        positive_class=exogenous_positive_class,
        num_splits=exogenous_positive_class,
        aggregate_func=aggregate_func,
    )
    d1_model.fit(X=exogenous_only_X_for_fit[["exogenous"]], y=y)

    assert d2_model.exogenous_thresholds_to_test == d1_model.threshold_to_test
    assert (d2_model._exogenous_splits == d1_model._splits).all()
    assert d2_model._endogenous_splits is None

    d2_y_pred = d2_model.predict(X_for_predict)
    d1_y_pred = d1_model.predict(X_for_predict[["exogenous"]])

    assert np.allclose(d2_y_pred, d1_y_pred)

    # Test for true 2-dimensional problem.

    d2_model = get_2d_model()
    X_for_fit = pd.DataFrame(data={
        "exogenous": exogenous_X_col_for_fit,
        "endogenous": endogenous_X_col_for_fit,
    })
    d2_model.fit(X=X_for_fit, y=y)

    assert d2_model._exogenous_splits is None or len(d2_model._exogenous_splits) == 2 * exogenous_num_splits + 1
    assert d2_model._endogenous_splits is None or len(d2_model._endogenous_splits) == 2 * endogenous_num_splits + 1

    d2_y_pred = d2_model.predict(X_for_predict)

    possible_exogenous_y_pred_values = (
        [] if d2_model._exogenous_splits is None
        else [i / len(d2_model._exogenous_splits) for i in range(len(d2_model._exogenous_splits))]
    )
    possible_endogenous_y_pred_values = (
        [] if d2_model._endogenous_splits is None
        else [i / len(d2_model._endogenous_splits) for i in range(len(d2_model._endogenous_splits))]
    )
    possible_y_pred_values = possible_exogenous_y_pred_values + possible_endogenous_y_pred_values
    if d2_model._exogenous_splits is not None and d2_model._endogenous_splits is not None:
        possible_y_pred_values.extend(list(
            (possible_exogenous_y_value + possible_endogenous_y_value) / 2
            for possible_exogenous_y_value, possible_endogenous_y_value
            in itertools.product(possible_exogenous_y_pred_values, possible_endogenous_y_pred_values)
        ))

    assert all([
        any(
            np.isclose(y_pred, possible_y_pred_value)
            for possible_y_pred_value
            in possible_y_pred_values
        )
        for y_pred
        in d2_y_pred
    ])


def test_piecewisetransformation():
    model = PiecewiseLinearTransformation(num_splits=4, positive_class=1)

    X = pd.DataFrame(np.arange(-9, 10, 1).T)
    y = pd.Series((np.arange(-9, 10, 1).T) * 0.1)
    model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    preds = model.predict(X)

    inverse_model = PiecewiseLinearTransformation(num_splits=4, positive_class=0)
    inverse_model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)
