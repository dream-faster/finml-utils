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
    df1 = pd.DataFrame({"first": [11, 12, None], "second": [5, 6, 7]}, index=[2, 3, 4])
    df2 = pd.DataFrame({"first": [1, None, 3, 4]}, index=[1, 2, 3, 4])

    df_last = concat_on_index_without_duplicates([df1, df2], keep="last")
    df_first = concat_on_index_without_duplicates([df1, df2], keep="first")

    assert len(df_last) == 4
    assert (
        ~(
            df_last["second"].replace(np.nan, None).values
            == np.array([None, 5.0, 6.0, 7.0])
        )
    ).sum() == 0
    assert (
        ~(
            df_last["first"].replace(np.nan, None).values
            == np.array([1.0, 11.0, 3.0, 4.0])
        )
    ).sum() == 0
    assert df_last.columns.to_list() == ["first", "second"]
    assert isinstance(df_last, pd.DataFrame)

    assert len(df_first) == 4
    assert (
        ~(
            df_first["second"].replace(np.nan, None).values
            == np.array([None, 5.0, 6.0, 7.0])
        )
    ).sum() == 0
    assert (
        ~(
            df_first["first"].replace(np.nan, None).values
            == np.array([1.0, 11.0, 12.0, 4.0])
        )
    ).sum() == 0
    assert df_first.columns.to_list() == ["first", "second"]
    assert isinstance(df_first, pd.DataFrame)

    ds_1 = pd.Series([11, 12, None], index=[2, 3, 4])
    ds_2 = pd.Series([1, None, 3, 4], index=[1, 2, 3, 4])
    ds_last = concat_on_index_without_duplicates([ds_1, ds_2], keep="last")
    ds_first = concat_on_index_without_duplicates([ds_1, ds_2], keep="first")

    assert isinstance(ds_last, pd.Series)
    assert isinstance(ds_first, pd.Series)
    assert ds_last.isna().sum() == 0
    assert ds_first.isna().sum() == 0
