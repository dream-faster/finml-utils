import pickle
from pathlib import Path


class ExtendedPath(Path):
    def add_suffix(self, suffix):
        return Path(str(self) + suffix)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".pkl").open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.with_suffix(".pkl").open("rb") as f:
        return pickle.load(f)
