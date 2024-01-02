import json
from pathlib import Path


def json_load(path: Path):
    with Path.open(path) as f:
        return json.load(f)


def json_save(obj, path: Path):
    with Path.open(path, "w") as f:
        json.dump(obj, f)
