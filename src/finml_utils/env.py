import os
import warnings

from dotenv import load_dotenv


def get_env(key: str) -> str | None:
    try:
        load_dotenv()
        return os.environ.get(key)
    except:  # noqa
        warnings.warn(f"Couldn't load {key}")  # noqa
        return None
