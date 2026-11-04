"""The demo codec must hand the in-browser engine exactly the inputs the server resolved."""

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
from fda_demo.codec import decode, encode


def roundtrip(obj: object) -> object:
    return decode(json.loads(json.dumps(encode(obj))))


def test_frames_and_series_roundtrip() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B", name="date")
    frame = pd.DataFrame(
        {
            "close": [1.0, np.nan, 3.5, float("inf"), 5.0],
            "volume": np.arange(5, dtype="int64"),
            "flag": [True, False, True, True, False],
            "label": ["a", "b", None, "d", "e"],
        },
        index=idx,
    )
    # an index frequency is not carried (lake series have none)
    pd.testing.assert_frame_equal(roundtrip(frame), frame, check_freq=False)
    series = pd.Series([0.1, -0.2, 0.3], index=pd.Index(["x", "y", "z"]), name="r")
    pd.testing.assert_series_equal(roundtrip(series), series)


def test_multiindex_and_tz_roundtrip() -> None:
    mi = pd.MultiIndex.from_tuples([("SPY", 1), ("SPY", 2), ("QQQ", 1)], names=["s", "n"])
    s = pd.Series([1.0, 2.0, 3.0], index=mi)
    pd.testing.assert_series_equal(roundtrip(s), s)
    tz = pd.Series([1, 2], index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"))
    pd.testing.assert_series_equal(roundtrip(tz), tz)


def test_scalars_and_containers_roundtrip() -> None:
    value = {
        "d": date(2024, 3, 1),
        "dt": datetime(2024, 3, 1, 12, 30),
        "td": pd.Timedelta(days=2),
        "ts": pd.Timestamp("2024-03-01"),
        "nat": pd.NaT,
        "tuple": (1, "a"),
        "set": {3},
        "nested": [{"x": np.float64(1.5)}],
    }
    back = roundtrip(value)
    assert isinstance(back, dict)
    assert back["d"] == value["d"] and back["dt"] == value["dt"] and back["td"] == value["td"]
    assert back["ts"] == value["ts"] and back["nat"] is pd.NaT
    assert back["tuple"] == (1, "a") and back["set"] == {3}
    assert back["nested"] == [{"x": 1.5}]
