import numpy as np
import pandas as pd
from finml_utils.dataframes import concat_on_index_without_duplicates, trim_initial_nans


def test_trim_initial_nans():
    X = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 3, 4, 5],
            "b": [np.nan, np.nan, np.nan, 4, 5],
        }
    )

    assert trim_initial_nans(X).equals(X.iloc[2:])

    X = pd.DataFrame(
        {
            "a": [0.0, 0.0, 3, 4, 5],
            "b": [np.nan, np.nan, 0.1, 4, 5],
        }
    )

    assert trim_initial_nans(X).equals(X)


def test_concat_on_index_without_duplicates():
    ds_1 = pd.Series([11, 12, None])
    ds_2 = pd.Series([1, None, 3, 4])
    df1 = pd.DataFrame({"first": ds_1.values, "second": [5, 6, 7]}, index=[2, 3, 4])
    df2 = pd.DataFrame({"first": ds_2.values}, index=[1, 2, 3, 4])

    df = concat_on_index_without_duplicates([df1, df2], keep="last")

    assert len(df) == 4
    assert list(df["first"].values) == [1, 11, 3, 4]
    assert list(df["second"].values) == [np.nan, 5.0, 6.0, 7.0]
    assert df.columms == ["first", "second"]
    assert isinstance(df, pd.DataFrame)

    ds = concat_on_index_without_duplicates([ds_1, ds_2], keep="last")

    assert isinstance(ds, pd.Series)
