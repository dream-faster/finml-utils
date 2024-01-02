from random import sample

import numpy as np
import pandas as pd


def shuffle_df_in_chunks(
    df: pd.DataFrame,
    chunk_size: int | float,
) -> pd.DataFrame:
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
    df.sort_index(inplace=True)
    df.index = original_index

    return df
