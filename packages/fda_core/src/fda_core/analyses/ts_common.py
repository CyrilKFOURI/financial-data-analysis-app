"""Time-Series Lab foundations (M06 §2, catalog TS-024): series selection & transformations,
and the shared test-result envelope (H0, statistic, p-value, critical values, reading).

Every TS- analysis takes a `series` choice; pair transformations (ratio, spread) use the
second subject resolved into `data["other_prices"]` (param `other`, default: the type's
benchmark).
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.analyses.price import _field, _prices, _windowed
from fda_core.registry import AnalysisInputs, InsufficientDataError, NotApplicableError

SeriesKind = Literal[
    "log_price", "price", "returns", "log_returns", "abs_returns", "squared_returns",
    "volume", "log_volume", "zscore", "detrended", "frac_diff", "ratio", "spread",
]  # fmt: skip

LEVEL_KINDS = frozenset(
    {"log_price", "price", "zscore", "detrended", "frac_diff", "ratio", "spread"}
)


class SeriesParams(BaseModel):
    series: SeriesKind = "returns"
    frac_d: float = Field(0.4, ge=0.0, le=1.0, description="d for the frac_diff transformation")
    other: str | None = Field(
        None, max_length=20, pattern=r"^[\w.^=-]+$", description="Second asset for ratio/spread"
    )


# ── fractional differentiation (AFML ch. 5) ──────────────────────────────────


def ffd_weights(d: float, threshold: float = 1e-5, max_len: int = 10_000) -> np.ndarray:
    """Fixed-width window weights w_k = −w_{k−1}(d − k + 1)/k until |w_k| < threshold."""
    weights = [1.0]
    k = 1
    while k < max_len:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights)


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """FFD transform; the first len(w) − 1 values (incomplete window) are dropped."""
    w = ffd_weights(d, threshold)
    width = len(w)
    values = series.to_numpy(float)
    if len(values) < width:
        raise InsufficientDataError(f"Frac-diff with d={d} needs ≥ {width} observations")
    out = np.convolve(values, w, mode="valid")  # Σ w_k x_{t−k}
    return pd.Series(out, index=series.index[width - 1 :])


# ── series selection ─────────────────────────────────────────────────────────


def _other_close(inputs: AnalysisInputs, index: pd.Index) -> pd.Series:
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError("This transformation needs a second asset with stored prices")
    return other[_field(inputs)].reindex(index).dropna()


def select_series(inputs: AnalysisInputs, params: SeriesParams, full: bool = False) -> pd.Series:
    """The series an analysis runs on, over the context range (or the full history)."""
    frame = _prices(inputs)
    if not full:
        frame = _windowed(inputs, frame)
    price = frame[_field(inputs)].astype(float)
    kind = params.series
    if kind in {"volume", "log_volume"}:
        if not inputs.conventions.get("has_volume", True):
            raise NotApplicableError("No meaningful volume for this asset type")
        volume = frame["volume"].astype(float)
        return np.log(volume[volume > 0]) if kind == "log_volume" else volume
    log_price = np.log(price)
    match kind:
        case "price":
            return price
        case "log_price":
            return log_price
        case "returns":
            return price.pct_change().dropna()
        case "log_returns":
            return log_price.diff().dropna()
        case "abs_returns":
            return price.pct_change().dropna().abs()
        case "squared_returns":
            return price.pct_change().dropna() ** 2
        case "zscore":
            return (log_price - log_price.mean()) / log_price.std(ddof=1)
        case "detrended":
            t = np.arange(len(log_price), dtype=float)
            slope, intercept = np.polyfit(t, log_price.to_numpy(), 1)
            return log_price - (intercept + slope * t)
        case "frac_diff":
            return frac_diff(log_price, params.frac_d)
        case "ratio" | "spread":
            other = _other_close(inputs, price.index)
            aligned = log_price.loc[other.index]
            if kind == "ratio":
                return aligned - np.log(other)
            beta = np.polyfit(np.log(other).to_numpy(), aligned.to_numpy(), 1)[0]
            return aligned - beta * np.log(other)
    raise NotApplicableError(f"Unknown series {kind}")


def require(series: pd.Series | np.ndarray, minimum: int, what: str) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < minimum:
        raise InsufficientDataError(f"{what} needs ≥ {minimum} observations, got {len(values)}")
    return values


# ── test envelope ────────────────────────────────────────────────────────────


def make_test(
    name: str,
    h0: str,
    statistic: float | None,
    p_value: float | None = None,
    critical_values: dict[str, float] | None = None,
    alpha: float = 0.05,
    reject: bool | None = None,
    reading_reject: str = "",
    reading_keep: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Uniform test row: decision from the p-value, else from the critical value at α."""
    if reject is None and p_value is not None and math.isfinite(p_value):
        reject = p_value < alpha
    if reject is not None:
        reject = bool(reject)  # numpy bools are not JSON-serialisable
    return {
        "test": name,
        "h0": h0,
        "statistic": None
        if statistic is None or not math.isfinite(statistic)
        else float(statistic),
        "p_value": None if p_value is None or not math.isfinite(p_value) else float(p_value),
        "critical_values": critical_values or {},
        "alpha": alpha,
        "reject": reject,
        "reading": (reading_reject if reject else reading_keep) if reject is not None else "",
        **(extra or {}),
    }


def iso(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def clean(values: Any) -> list[float | None]:
    return [
        None if v is None or not np.isfinite(v) else float(v) for v in np.asarray(values, float)
    ]


def sample_info(series: pd.Series, params: SeriesParams) -> dict[str, Any]:
    return {
        "series": params.series,
        "n": len(series),
        "from": iso(series.index[:1])[0] if len(series) else None,
        "to": iso(series.index[-1:])[0] if len(series) else None,
    }
