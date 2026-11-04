"""JSON codec for analysis inputs: what the API resolver hands to an analysis (frames, series,
dates, dataclasses, pydantic models, numpy arrays…) written once at export time and read back in
the browser. Tagged objects keep dtypes and index types across pandas versions (the browser
runs Pyodide's pandas, the exporter the project's)."""

import dataclasses
import datetime as dt
import importlib
import math
from typing import Any

import numpy as np
import pandas as pd


def _float(x: float) -> Any:
    if math.isnan(x):
        return {"__f__": "nan"}
    if math.isinf(x):
        return {"__f__": "inf" if x > 0 else "-inf"}
    return x


def _array(values: Any) -> dict[str, Any]:
    """One column / index: a dtype tag and plain values."""
    arr = pd.array(values) if not isinstance(values, pd.api.extensions.ExtensionArray) else values
    dtype = str(getattr(values, "dtype", arr.dtype))
    if isinstance(getattr(values, "dtype", None), pd.DatetimeTZDtype):
        tz = str(values.dtype.tz)
        return {
            "__a__": "dt_tz",
            "tz": tz,
            "v": [None if pd.isna(x) else x.isoformat() for x in values],
        }
    if dtype.startswith("datetime64"):
        return {
            "__a__": "dt",
            "v": [None if pd.isna(x) else pd.Timestamp(x).isoformat() for x in values],
        }
    if dtype.startswith("timedelta64"):
        return {"__a__": "td", "v": [None if pd.isna(x) else pd.Timedelta(x).value for x in values]}
    if dtype == "category":
        return {"__a__": "cat", "v": [encode(x) for x in values.astype(object)]}
    if dtype == "bool":
        return {"__a__": "bool", "v": [bool(x) for x in values]}
    if dtype.startswith(("int", "uint")):
        return {"__a__": "int", "v": [int(x) for x in values]}
    if dtype.startswith("float"):
        return {"__a__": "float", "v": [None if x != x else _float(float(x)) for x in values]}
    if dtype.startswith(("Int", "UInt", "Float", "boolean")):
        return {
            "__a__": "nullable",
            "dtype": dtype,
            "v": [None if pd.isna(x) else encode(x) for x in values],
        }
    return {"__a__": "obj", "v": [encode(x) for x in pd.Series(values, dtype=object)]}


def _unarray(a: dict[str, Any]) -> Any:
    kind, v = a["__a__"], a["v"]
    if kind == "dt":
        return pd.to_datetime(pd.Series(v, dtype=object)).to_numpy()
    if kind == "dt_tz":
        return pd.DatetimeIndex(pd.to_datetime(pd.Series(v, dtype=object), utc=True)).tz_convert(
            a["tz"]
        )
    if kind == "td":
        return pd.to_timedelta(pd.Series(v, dtype="float64"), unit="ns").to_numpy()
    if kind == "cat":
        return pd.Categorical([decode(x) for x in v])
    if kind == "bool":
        return np.array(v, dtype=bool)
    if kind == "int":
        return np.array(v, dtype="int64")
    if kind == "float":
        return np.array([math.nan if x is None else decode(x) for x in v], dtype="float64")
    if kind == "nullable":
        return pd.array([None if x is None else decode(x) for x in v], dtype=a["dtype"])
    return np.array([decode(x) for x in v] + [None], dtype=object)[:-1]  # keeps tuples whole


def _index(idx: pd.Index) -> dict[str, Any]:
    if isinstance(idx, pd.MultiIndex):
        return {
            "__mi__": [_index(idx.get_level_values(i)) for i in range(idx.nlevels)],
            "names": [encode(n) for n in idx.names],
        }
    if isinstance(idx, pd.RangeIndex):
        return {"__ri__": [idx.start, idx.stop, idx.step], "name": encode(idx.name)}
    return {"__ix__": _array(idx), "name": encode(idx.name)}


def _unindex(d: dict[str, Any]) -> pd.Index:
    if "__mi__" in d:
        levels = [_unindex(x) for x in d["__mi__"]]
        return pd.MultiIndex.from_arrays(levels, names=[decode(n) for n in d["names"]])
    if "__ri__" in d:
        start, stop, step = d["__ri__"]
        return pd.RangeIndex(start, stop, step, name=decode(d["name"]))
    return pd.Index(_unarray(d["__ix__"]), name=decode(d["name"]))


def encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _float(obj)
    if isinstance(obj, np.generic):
        return encode(obj.item())
    if isinstance(obj, pd.DataFrame):
        return {
            "__df__": [_array(obj.iloc[:, i]) for i in range(obj.shape[1])],
            "columns": _index(obj.columns),
            "index": _index(obj.index),
            "attrs": encode(dict(obj.attrs)),
        }
    if isinstance(obj, pd.Series):
        return {"__s__": _array(obj), "index": _index(obj.index), "name": encode(obj.name)}
    if isinstance(obj, pd.Index):
        return _index(obj)
    if isinstance(obj, pd.Timestamp):
        return {"__ts__": obj.isoformat()} if not pd.isna(obj) else {"__nat__": True}
    if isinstance(obj, pd.Timedelta):
        return {"__td__": obj.value}
    if obj is pd.NaT:
        return {"__nat__": True}
    if isinstance(obj, dt.datetime):
        return {"__dtm__": obj.isoformat()}
    if isinstance(obj, dt.date):
        return {"__d__": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {"__nd__": _array(obj.reshape(-1)), "shape": list(obj.shape)}
    if isinstance(obj, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in obj):
            return {k: encode(v) for k, v in obj.items()}
        return {"__map__": [[encode(k), encode(v)] for k, v in obj.items()]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, tuple):
        return {"__t__": [encode(x) for x in obj]}
    if isinstance(obj, (set, frozenset)):
        return {
            "__set__": [encode(x) for x in sorted(obj, key=repr)],
            "frozen": isinstance(obj, frozenset),
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        fields = {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return {"__dc__": f"{cls.__module__}:{cls.__qualname__}", "f": fields}
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        cls = type(obj)
        return {"__pm__": f"{cls.__module__}:{cls.__qualname__}", "d": encode(model_dump())}
    raise TypeError(f"Cannot encode {type(obj).__module__}.{type(obj).__qualname__}")


def _class(path: str) -> Any:
    module, _, qualname = path.partition(":")
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    if "__f__" in obj:
        return float(obj["__f__"])
    if "__df__" in obj:
        columns = _unindex(obj["columns"])
        index = _unindex(obj["index"])
        frame = pd.DataFrame({i: _unarray(c) for i, c in enumerate(obj["__df__"])}, index=index)
        frame.columns = columns
        frame.attrs.update(decode(obj["attrs"]) or {})
        return frame
    if "__s__" in obj:
        return pd.Series(
            _unarray(obj["__s__"]), index=_unindex(obj["index"]), name=decode(obj["name"])
        )
    if "__ix__" in obj or "__mi__" in obj or "__ri__" in obj:
        return _unindex(obj)
    if "__ts__" in obj:
        return pd.Timestamp(obj["__ts__"])
    if "__nat__" in obj:
        return pd.NaT
    if "__td__" in obj:
        return pd.Timedelta(obj["__td__"], unit="ns")
    if "__dtm__" in obj:
        return dt.datetime.fromisoformat(obj["__dtm__"])
    if "__d__" in obj:
        return dt.date.fromisoformat(obj["__d__"])
    if "__nd__" in obj:
        return np.asarray(_unarray(obj["__nd__"])).reshape(obj["shape"])
    if "__map__" in obj:
        return {decode(k): decode(v) for k, v in obj["__map__"]}
    if "__t__" in obj:
        return tuple(decode(x) for x in obj["__t__"])
    if "__set__" in obj:
        items = [decode(x) for x in obj["__set__"]]
        return frozenset(items) if obj.get("frozen") else set(items)
    if "__dc__" in obj:
        cls = _class(obj["__dc__"])
        return cls(**{k: decode(v) for k, v in obj["f"].items()})
    if "__pm__" in obj:
        return _class(obj["__pm__"]).model_validate(decode(obj["d"]))
    return {k: decode(v) for k, v in obj.items()}
