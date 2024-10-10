import numpy as np
from finml_utils.piecewisetransformation import PiecewiseLinearTransformation


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
