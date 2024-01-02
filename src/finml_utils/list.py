from collections.abc import Iterable
from typing import TypeVar

import pandas as pd
from krisi.utils.data import shuffle_df_in_chunks


def flatten_iterable(input: list[Iterable] | Iterable) -> list:
    def _flatten_iterable(input: list[Iterable] | Iterable) -> Iterable:
        for x in input:
            if isinstance(x, (list, tuple)):
                yield from _flatten_iterable(x)
            else:
                yield x

    return list(_flatten_iterable(input))


T = TypeVar("T", pd.DataFrame, pd.Series)


def shuffle_but_keep_some_in_tact(
    df: T,
    chunk_size: float | int,
    fraction_to_keep_in_tact: float,
) -> T:
    df_intact = df.sample(frac=fraction_to_keep_in_tact)
    shuffled = shuffle_df_in_chunks(df, chunk_size)
    shuffled[df_intact.index] = df_intact
    return shuffled


def merge_small_chunk(
    chunks: list[tuple[int, int]],
    min_chunk_size: int,
) -> list[tuple[int, int]]:
    for i, chunk in enumerate(chunks):
        if chunk[1] - chunk[0] < min_chunk_size:
            chunks.remove(chunk)
            chunks[i - 1] = (chunks[i - 1][0], chunk[1])
            break
    return chunks


T = TypeVar("T")


def difference(a: list[T], b: list[T]) -> list[T]:
    b = set(b)  # type: ignore
    return [aa for aa in a if aa not in b]
