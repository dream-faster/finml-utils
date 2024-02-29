import collections
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from iteration_utilities import unique_everseen

T = TypeVar("T")


def flatten_iterable(iterable: Sequence[Iterable] | Iterable) -> list:
    def _flatten_iterable(iterable: Sequence[Iterable] | Iterable) -> Iterable:
        for x in iterable:
            if isinstance(x, list | tuple):
                yield from _flatten_iterable(x)
            else:
                yield x

    return list(_flatten_iterable(iterable))


def group_by(input_list: Sequence[T], key_extractor: Callable) -> dict[str, list[T]]:
    result = {}
    for item in input_list:
        key = key_extractor(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def merge_small_chunk(
    chunks: Sequence[tuple[int, int]],
    min_chunk_size: int,
) -> list[tuple[int, int]]:
    for i, chunk in enumerate(chunks):
        if chunk[1] - chunk[0] < min_chunk_size:
            chunks.remove(chunk)
            chunks[i - 1] = (chunks[i - 1][0], chunk[1])
            break
    return chunks


def difference(a: Sequence[T], b: Sequence[T]) -> list[T]:
    b = set(b)  # type: ignore
    return [aa for aa in a if aa not in b]


def wrap_in_list(item: T | Sequence[T]) -> list[T]:
    return item if isinstance(item, list) else [item]


def transform_range_to_list(input_range: range | Sequence[T]) -> list[T]:
    return list(input_range) if isinstance(input_range, range) else input_range


def wrap_in_double_list_if_needed(
    input_list: T | Sequence[T],
) -> list[list[T]] | list[T]:
    """
    If input is a single item, wrap it in a list.
    If input is a single list, wrap it in another list.
    If input is a list of lists, return it as is.
    """
    if not isinstance(input_list, list):
        return [input_list]
    if isinstance(input_list[0], list):
        return input_list
    return [input_list]


def flatten(input_list: Sequence[list] | list) -> Iterable:
    for x in input_list:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x


def flatten_lists(input_list: Sequence[list] | list) -> list:
    return list(flatten(input_list))


def keep_only_duplicates(input_list: list) -> list:
    return [
        item for item, count in collections.Counter(input_list).items() if count > 1
    ]


def has_intersection(lhs: list, rhs: list) -> bool:
    return len(set(lhs).intersection(rhs)) > 0


def unique(input_list: list) -> list:
    return unique_everseen(input_list)


def swap_tuples(input_list: Sequence[tuple]) -> Sequence[tuple]:
    return [(b, a) for a, b in input_list]


def filter_none(input_list: list) -> list:
    return [x for x in input_list if x is not None]


def empty_if_none(input_list: list | None) -> list:
    return [] if input_list is None else input_list


def unpack_list_of_tuples(input_list: Sequence[tuple]):
    if len(input_list) == 1:
        return [[item] for item in input_list[0]]
    return zip(*input_list, strict=False)
