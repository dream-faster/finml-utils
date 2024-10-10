from __future__ import annotations

import json
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


def pandas_to_json(ds: pd.Series | pd.DataFrame) -> dict:
    if isinstance(ds, pd.Series) and ds.name is None:
        ds = ds.rename("")
    if ds.empty:
        return dict(
            columns=ds.columns.to_list() if isinstance(ds, pd.DataFrame) else [ds.name],
            data=[],
            index=[],
            name="",
        )
    return json.loads(
        ds.round(4).to_json(date_format="iso", orient="split", date_unit="s")
    )


def to_json_if_encodable(obj: object) -> object:
    return obj.to_json() if isinstance(obj, JSONEncodable) else obj


class JSONEncodableEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, pd.Series | pd.DataFrame):
            if isinstance(o.index, pd.DatetimeIndex):
                # return pandas_to_json(o.resample("W").last())
                return pandas_to_json(o)
            return pandas_to_json(o)
        if isinstance(o, JSONEncodable):
            return o.to_json()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, np.number):
            return o.item()

        return super().default(o)


class JSONEncodable:
    def to_json(self) -> dict:
        return {k: to_json_if_encodable(getattr(self, k)) for k in self.__dict__}
