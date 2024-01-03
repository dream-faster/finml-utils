from random import sample
from typing import TypeVar

import numpy as np
import pandas as pd

TPandas = TypeVar("TPandas", pd.DataFrame, pd.Series)


def shuffle_in_chunks(
    df: TPandas,
    chunk_size: int | float,
) -> TPandas:
    """Shuffles a dataframe by rows in chunks.

    Parameters
    ----------
    df : pd.DataFrame
        Data, whose rows should be shuffled in chunks
    chunk_size : Union[int, float]
        Either a fixed chunk size or a fraction of the total number of rows

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame by rows in chunks.
    """
    original_index = df.index
    df = df.reset_index(drop=True)
    chunk_size = (
        int(chunk_size * len(df)) if isinstance(chunk_size, float) else chunk_size
    )
    num_rows = df.shape[0]
    num_chunks = num_rows // chunk_size
    shuffled_idx = []
    for i in sample(list(range(num_chunks)), num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        shuffled_idx.append(df.index[start_idx:end_idx])

    # If there are any remaining rows, shuffle and append them as well
    if num_rows % chunk_size != 0:
        shuffled_idx.append(df.index[num_chunks * chunk_size :])

    df = df.copy()
    df.index = df.index[np.concatenate(shuffled_idx)]
    df.sort_index(inplace=True)  # noqa: PD002
    df.index = original_index

    return df


def partial_shuffle_in_chunks(
    df: TPandas,
    chunk_size: float | int,
    fraction_to_keep_intact: float,
) -> TPandas:
    df_intact = df.sample(frac=fraction_to_keep_intact, replace=False)
    shuffled = shuffle_in_chunks(df, chunk_size)
    shuffled[df_intact.index] = df_intact
    return shuffled
