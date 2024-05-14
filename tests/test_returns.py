import numpy as np
import pandas as pd
from finml_utils.returns import to_log_returns, to_returns


def test_returns_clip():
    data = pd.Series([0.0, 0.0, 1.3, 1.4, 3.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    assert not np.isinf(to_returns(data, clip=1.0)).any()


def test_log_returns_clip():
    data = pd.Series([0.0, 0.0, 1.3, 1.4, 3.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    assert not np.isinf(to_log_returns(data, clip=1.0)).any()
