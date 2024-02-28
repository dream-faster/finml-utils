from collections.abc import Callable, Sequence
from typing import TypeVar

from pathos.multiprocessing import Pool, cpu_count

T = TypeVar("T")
R = TypeVar("R")


def pmap(
    func: Callable[[T], R],
    iterable: Sequence[T],
    n_jobs: int = -1,
    pool: Pool | None = None,
) -> list[R]:
    assert n_jobs != 0
    n_cpus = cpu_count() if n_jobs == -1 else n_jobs
    pool = pool or Pool(n_cpus)
    return (
        pool.map(
            func,
            iterable,
            chunksize=max(1, int(len(iterable) // n_cpus)),
        )
        if len(iterable) > 2 and n_cpus > 1  # noqa: PLR2004
        else [func(el) for el in iterable]
    )
