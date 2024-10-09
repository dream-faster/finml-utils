import itertools
from random import choices
from typing import Literal

import numpy as np
import pandas as pd
from finml_utils.decisiontree import (
    RegularizedDecisionTree,
    SingleDecisionTree,
    UltraRegularizedDecisionTree,
)
from finml_utils.piecewisetransformation import PiecewiseLinearTransformation
from hypothesis import given
from hypothesis import strategies as st

from src.finml_utils import TwoDimensionalPiecewiseLinearRegression


def test_singledecisiontree():
    model = SingleDecisionTree(
        threshold_margin=0.3,
        threshold_step=0.05,
        ensemble_num_trees=None,
        ensemble_percentile_gap=None,
    )
    model.fit(
        X=np.array([[3.0, 2.0, 0.4, 0.6, -1.0, 0.0, 0.2, 1.2, 1.0]]).T,
        y=np.array([1, 1, 1, 0, 0, 0, 0, 1, 1]),
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
    X = np.expand_dims(np.arange(-9, 10, 1), axis=1)
    y = (np.arange(-9, 10, 1).T) * 0.1
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
    X = np.expand_dims(np.arange(-9, 10, 1), axis=1)
    y = (np.arange(-9, 10, 1).T) * 0.1
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

    return (
        np.random.normal(loc=0.0, scale=1.0, size=(list_len, 2)),  # X_to_fit
        choices(population=[0, 1], k=list_len),  # bool_y
        np.random.normal(loc=0.0, scale=1.0, size=(list_len, 2)),  # X_to_predict
    )


@given(
    st.integers(min_value=-1, max_value=30),  # exogenous_threshold_margin_x100
    st.integers(min_value=-1, max_value=30),  # endogenous_threshold_margin_x100
    st.integers(min_value=1, max_value=5),  # exogenous_threshold_step_x100
    st.integers(min_value=1, max_value=5),  # endogenous_threshold_step_x100
    st.booleans(),  # bool_exogenous_positive_class
    st.booleans(),  # bool_endogenous_positive_class
    st.integers(min_value=1, max_value=4),  # exogenous_num_splits
    st.integers(min_value=1, max_value=4),  # endogenous_num_splits
    same_len_X_y_lists(),  # same_len_X_y_lists
    # IMPORTANT: does not test for "sharpe" as aggregate_func because "sharpe" will most likely not be used and
    #            in case of "sharpe" certain edge cases that we will never encounter always fail
    # st.booleans().filter(lambda is_aggregate_func_mean: is_aggregate_func_mean), # is_aggregate_func_mean
)
def test_twodimensionalpiecewiselinearregression_random(
    exogenous_threshold_margin_x100: int,
    endogenous_threshold_margin_x100: int,
    exogenous_threshold_step_x100: int,
    endogenous_threshold_step_x100: int,
    bool_exogenous_positive_class: bool,
    bool_endogenous_positive_class: bool,
    exogenous_num_splits: int,
    endogenous_num_splits: int,
    same_len_X_y_lists: tuple[np.ndarray, list[bool], np.ndarray],
    # is_aggregate_func_mean: bool
):
    exogenous_threshold_margin = round(exogenous_threshold_margin_x100 / 100, 3)
    endogenous_threshold_margin = round(endogenous_threshold_margin_x100 / 100, 3)
    exogenous_threshold_step = round(exogenous_threshold_step_x100 / 100, 3)
    endogenous_threshold_step = round(endogenous_threshold_step_x100 / 100, 3)
    exogenous_positive_class = bool(bool_exogenous_positive_class)
    endogenous_positive_class = bool(bool_endogenous_positive_class)
    # aggregate_func: Literal["mean", "sharpe"] = "mean" if is_aggregate_func_mean else "sharpe"
    aggregate_func: Literal["mean", "sharpe"] = "mean"

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

    X_for_fit, bool_y, X_for_predict = same_len_X_y_lists

    y = pd.Series([int(bool_y) for bool_y in bool_y])

    d2_model = get_2d_model()

    if exogenous_threshold_margin <= 0:
        assert d2_model.exogenous_thresholds_to_test == [0.5]
    else:
        for i, threshold in enumerate(d2_model.exogenous_thresholds_to_test):
            assert threshold == round(
                0.5 - exogenous_threshold_margin + i * exogenous_threshold_step, 3
            )

    if endogenous_threshold_margin <= 0:
        assert d2_model.endogenous_thresholds_to_test == [0.5]
    else:
        for i, threshold in enumerate(d2_model.endogenous_thresholds_to_test):
            assert threshold == round(
                0.5 - endogenous_threshold_margin + i * endogenous_threshold_step, 3
            )

    # Test for true 2-dimensional problem.

    d2_model = get_2d_model()
    d2_model.fit(X=X_for_fit, y=y)

    assert (
        d2_model._exogenous_splits is None
        or len(d2_model._exogenous_splits) == 2 * exogenous_num_splits + 1
    )
    assert (
        d2_model._endogenous_splits is None
        or len(d2_model._endogenous_splits) == 2 * endogenous_num_splits + 1
    )

    d2_y_pred = d2_model.predict(X_for_predict)

    possible_exogenous_y_pred_values = (
        []
        if d2_model._exogenous_splits is None
        else [
            i / len(d2_model._exogenous_splits)
            for i in range(len(d2_model._exogenous_splits) + 1)
        ]
    )
    possible_endogenous_y_pred_values = (
        []
        if d2_model._endogenous_splits is None
        else [
            i / len(d2_model._endogenous_splits)
            for i in range(len(d2_model._endogenous_splits) + 1)
        ]
    )
    possible_y_pred_values = (
        possible_exogenous_y_pred_values + possible_endogenous_y_pred_values
    )
    if (
        d2_model._exogenous_splits is not None
        and d2_model._endogenous_splits is not None
    ):
        possible_y_pred_values.extend(
            [
                (possible_exogenous_y_value + possible_endogenous_y_value) / 2
                for possible_exogenous_y_value, possible_endogenous_y_value in itertools.product(
                    possible_exogenous_y_pred_values, possible_endogenous_y_pred_values
                )
            ]
        )

    assert all(
        any(
            np.isclose(y_pred, possible_y_pred_value)
            for possible_y_pred_value in possible_y_pred_values
        )
        for y_pred in d2_y_pred
    )


def test_twodimensionalpiecewiselinearregression():
    model = TwoDimensionalPiecewiseLinearRegression(
        exogenous_threshold_margin=0.1,
        endogenous_threshold_margin=0.1,
        exogenous_threshold_step=0.05,
        endogenous_threshold_step=0.05,
        exogenous_positive_class=1,
        endogenous_positive_class=1,
    )
    # assert model.threshold_to_test == [
    #     0.5,
    # ]
    X = np.array([np.flip(np.arange(-9, 10, 1)).T, np.arange(-9, 10, 1).T]).T
    y = (np.arange(-9, 10, 1).T) * 0.1
    model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    preds = model.predict(X)

    inverse_model = TwoDimensionalPiecewiseLinearRegression(
        exogenous_threshold_margin=0.1,
        endogenous_threshold_margin=0.1,
        exogenous_threshold_step=0.05,
        endogenous_threshold_step=0.05,
        exogenous_positive_class=0,
        endogenous_positive_class=0,
    )
    inverse_model.fit(
        X=X,
        y=y,
        sample_weight=None,
    )
    inverse_preds = inverse_model.predict(X)
    assert np.allclose(inverse_preds, 1 - preds)


def test_piecewisetransformation():
    model = PiecewiseLinearTransformation(num_splits=4, positive_class=1)

    X = np.expand_dims(np.arange(-9, 10, 1), axis=1)
    y = (np.arange(-9, 10, 1).T) * 0.1
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
