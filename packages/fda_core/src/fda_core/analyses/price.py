"""Price & trend analyses (ANALYTICS_CATALOG T-001…T-013, T-030; module M04).

Inputs (resolved by the API): `data["prices"]` — daily bars indexed by session date with
open/high/low/close/adj_close/volume — and `conventions` from the asset type
(`annualisation`, `price_field`, `has_volume`, `range`). Performance uses `price_field`
(adjusted close for equities/ETFs). Below `min_obs` an analysis returns null + reason.
"""

import math
from datetime import date, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.registry import (
    AnalysisInputs,
    AnalysisOutput,
    InsufficientDataError,
    NotApplicableError,
    SubjectKind,
    analysis,
)

PRICE_SUBJECTS: set[SubjectKind] = {"asset", "portfolio"}  # portfolios via their TWR NAV
DAYS_PER_YEAR = 365.25

RANGE_DAYS: dict[str, int] = {
    "1M": 31, "3M": 92, "6M": 183, "1Y": 365, "3Y": 3 * 365 + 1, "5Y": 5 * 365 + 1,
    "10Y": 10 * 365 + 2,
}  # fmt: skip


# ── helpers ──────────────────────────────────────────────────────────────────


def _prices(inputs: AnalysisInputs) -> pd.DataFrame:
    frame = inputs.data.get("prices")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise InsufficientDataError("No price history stored for this asset")
    return frame


def _field(inputs: AnalysisInputs) -> str:
    return str(inputs.conventions.get("price_field", "adj_close"))


def _annualisation(inputs: AnalysisInputs) -> int:
    factor = inputs.conventions.get("annualisation")
    if factor is None:
        raise NotApplicableError("Levels series (rates, macro): use level changes, not returns")
    return int(factor)


def window_start(range_code: str, last: date, first: date) -> date:
    """Start date of the visible range ending at `last` (YTD = first session of the year)."""
    if range_code == "MAX":
        return first
    if range_code == "YTD":
        return date(last.year, 1, 1)
    return last - timedelta(days=RANGE_DAYS.get(range_code, 365))


def _windowed(inputs: AnalysisInputs, frame: pd.DataFrame) -> pd.DataFrame:
    range_code = str(inputs.conventions.get("range", "1Y"))
    last, first = frame.index[-1], frame.index[0]
    return frame.loc[frame.index >= window_start(range_code, last, first)]


def _require(n: int, minimum: int, what: str) -> None:
    if n < minimum:
        raise InsufficientDataError(f"{what} needs ≥ {minimum} observations, got {n}")


def cagr(start_value: float, end_value: float, days: int) -> float:
    """T-005: (V_T / V_0)^(365.25 / days) − 1."""
    return float((end_value / start_value) ** (DAYS_PER_YEAR / days) - 1)


def annualised_volatility(returns: np.ndarray, annualisation: int) -> float:
    """T-007: sample standard deviation (ddof = 1) × √A."""
    return float(np.std(returns, ddof=1) * math.sqrt(annualisation))


def _dates(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def _clean(values: pd.Series | np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


# ── T-001 price chart ────────────────────────────────────────────────────────


class PriceChartParams(BaseModel):
    style: Literal["candles", "ohlc", "line", "area", "heikin_ashi"] = "candles"


@analysis(
    id="T-001",
    version="1.0.0",
    title="Price chart",
    description="Daily OHLCV over the context range; Heikin-Ashi smoothing on request.",
    applies_to=PRICE_SUBJECTS,
    params=PriceChartParams,
    result_kind="series",
    requires={"prices"},
    min_obs=2,
)
def price_chart(inputs: AnalysisInputs, params: PriceChartParams) -> AnalysisOutput:
    frame = _windowed(inputs, _prices(inputs))
    _require(len(frame), 2, "A chart")
    o, h, low, c = (frame[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if params.style == "heikin_ashi":
        ha_close = (o + h + low + c) / 4
        ha_open = np.empty_like(ha_close)
        ha_open[0] = (o[0] + c[0]) / 2
        for i in range(1, len(ha_open)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
        h, low = (
            np.maximum.reduce([h, ha_open, ha_close]),
            np.minimum.reduce([low, ha_open, ha_close]),
        )
        o, c = ha_open, ha_close
    has_volume = bool(inputs.conventions.get("has_volume", True))
    payload: dict[str, Any] = {
        "style": params.style,
        "dates": _dates(frame.index),
        "open": _clean(o),
        "high": _clean(h),
        "low": _clean(low),
        "close": _clean(c),
        "volume": _clean(frame["volume"]) if has_volume else None,
    }
    warnings = () if has_volume else ("No meaningful volume for this asset type.",)
    return AnalysisOutput("series", payload, warnings)


# ── T-002 base-100 ───────────────────────────────────────────────────────────


class NoParams(BaseModel):
    pass


@analysis(
    id="T-002",
    version="1.0.0",
    title="Base-100 performance",
    description="I_t = 100 · Π(1 + r_s), rebased at the start of the visible range.",
    applies_to=PRICE_SUBJECTS,
    params=NoParams,
    result_kind="series",
    requires={"prices"},
    min_obs=2,
)
def base_100(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    _annualisation(inputs)  # levels series are not rebased
    series = _windowed(inputs, _prices(inputs))[_field(inputs)]
    _require(len(series), 2, "Base-100")
    rebased = 100 * series / series.iloc[0]
    return AnalysisOutput("series", {"dates": _dates(series.index), "values": _clean(rebased)})


# ── T-003 period returns ─────────────────────────────────────────────────────

PERIODS = ("1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y", "ITD")
PERIOD_DAYS = {"1W": 7, "1M": 31, "3M": 92, "6M": 183} | {
    k: v for k, v in RANGE_DAYS.items() if k.endswith("Y")
}


def _anchor(series: pd.Series, period: str) -> tuple[date, float] | None:
    """Last close on or before the period start; None if history is too short."""
    index = series.index
    last = pd.Timestamp(index[-1]).date()
    if period == "1D":
        if len(series) < 2:
            return None
        return pd.Timestamp(index[-2]).date(), float(series.iloc[-2])
    if period == "ITD":
        return pd.Timestamp(index[0]).date(), float(series.iloc[0])
    if period == "MTD":
        start = date(last.year, last.month, 1) - timedelta(days=1)
    elif period == "YTD":
        start = date(last.year - 1, 12, 31)
    else:
        start = last - timedelta(days=PERIOD_DAYS[period])
    eligible = series.loc[series.index <= pd.Timestamp(start)]
    if eligible.empty:
        return None
    return pd.Timestamp(eligible.index[-1]).date(), float(eligible.iloc[-1])


@analysis(
    id="T-003",
    version="1.0.0",
    title="Period returns",
    description=(
        "Total return over 1D…10Y and since inception, measured from the last close on or "
        "before the period start; periods longer than one year are also annualised (T-005)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=NoParams,
    result_kind="table",
    requires={"prices"},
    min_obs=2,
)
def period_returns(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    _annualisation(inputs)
    series = _prices(inputs)[_field(inputs)]
    _require(len(series), 2, "Period returns")
    end_value = float(series.iloc[-1])
    last = pd.Timestamp(series.index[-1]).date()
    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        anchor = _anchor(series, period)
        if anchor is None:
            rows.append({"period": period, "return": None, "annualised": None,
                         "reason": "history shorter than the period"})  # fmt: skip
            continue
        start_date, start_value = anchor
        days = (last - start_date).days
        total = end_value / start_value - 1
        annualised = cagr(start_value, end_value, days) if days > 366 else None
        rows.append({"period": period, "from": start_date.isoformat(), "return": total,
                     "annualised": annualised})  # fmt: skip
    return AnalysisOutput("table", {"as_of": last.isoformat(), "rows": rows})


# ── T-005 CAGR / T-007 volatility ────────────────────────────────────────────


@analysis(
    id="T-005",
    version="1.0.0",
    title="CAGR",
    description="(V_T / V_0)^(365.25 / days) − 1 over the context range.",
    applies_to=PRICE_SUBJECTS,
    params=NoParams,
    result_kind="value",
    requires={"prices"},
    min_obs=2,
)
def compound_growth(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    _annualisation(inputs)
    series = _windowed(inputs, _prices(inputs))[_field(inputs)]
    _require(len(series), 2, "CAGR")
    days = (pd.Timestamp(series.index[-1]) - pd.Timestamp(series.index[0])).days
    _require(days, 30, "CAGR (calendar days)")
    value = cagr(float(series.iloc[0]), float(series.iloc[-1]), days)
    warnings = ("Annualised over less than one year: extrapolated.",) if days < 365 else ()
    return AnalysisOutput("value", {"value": value, "days": days}, warnings)


@analysis(
    id="T-007",
    version="1.0.0",
    title="Annualised volatility",
    description="Sample std of simple daily returns × √A (A from the asset type: 252/260/365).",
    applies_to=PRICE_SUBJECTS,
    params=NoParams,
    result_kind="value",
    requires={"prices"},
    min_obs=21,
)
def volatility(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    series = _windowed(inputs, _prices(inputs))[_field(inputs)]
    returns = series.pct_change().dropna().to_numpy(float)
    _require(len(returns), 20, "Volatility")
    return AnalysisOutput(
        "value",
        {"value": annualised_volatility(returns, factor), "n": len(returns),
         "annualisation": factor},
    )  # fmt: skip


# ── T-011 moving averages ────────────────────────────────────────────────────

MaKind = Literal["sma", "ema", "wma", "dema", "tema", "hma", "kama"]


def moving_average(values: pd.Series, kind: MaKind, period: int) -> pd.Series:
    """Catalog formulas: EMA α = 2/(n+1); DEMA = 2E₁ − E₂; TEMA = 3E₁ − 3E₂ + E₃;
    HMA = WMA(2·WMA(n/2) − WMA(n), √n); KAMA with ER over n, fast 2, slow 30."""

    def ema(x: pd.Series, n: int) -> pd.Series:
        return x.ewm(span=n, adjust=False, min_periods=n).mean()

    def wma(x: pd.Series, n: int) -> pd.Series:
        weights = np.arange(1, n + 1, dtype=float)
        return x.rolling(n).apply(lambda w: float(np.dot(w, weights) / weights.sum()), raw=True)

    match kind:
        case "sma":
            return values.rolling(period).mean()
        case "ema":
            return ema(values, period)
        case "wma":
            return wma(values, period)
        case "dema":
            e1 = ema(values, period)
            return 2 * e1 - ema(e1, period)
        case "tema":
            e1 = ema(values, period)
            e2 = ema(e1, period)
            return 3 * e1 - 3 * e2 + ema(e2, period)
        case "hma":
            half = wma(values, max(period // 2, 1))
            return wma(2 * half - wma(values, period), max(int(math.sqrt(period)), 1))
        case "kama":
            x = values.to_numpy(float)
            out = np.full(len(x), np.nan)
            fast, slow = 2 / (2 + 1), 2 / (30 + 1)
            if len(x) > period:
                out[period] = x[period]
                for i in range(period + 1, len(x)):
                    change = abs(x[i] - x[i - period])
                    noise = np.abs(np.diff(x[i - period : i + 1])).sum()
                    er = change / noise if noise else 0.0
                    sc = (er * (fast - slow) + slow) ** 2
                    out[i] = out[i - 1] + sc * (x[i] - out[i - 1])
            return pd.Series(out, index=values.index)


class MovingAverageParams(BaseModel):
    kind: MaKind = "sma"
    periods: list[int] = Field(default=[50, 200], min_length=1, max_length=6)
    price_field: Literal["close", "adj_close"] = "close"


@analysis(
    id="T-011",
    version="1.0.0",
    title="Moving averages",
    description="SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA on the chart price; warm-up on full history.",
    applies_to=PRICE_SUBJECTS,
    params=MovingAverageParams,
    result_kind="series",
    requires={"prices"},
    min_obs=2,
)
def moving_averages(inputs: AnalysisInputs, params: MovingAverageParams) -> AnalysisOutput:
    full = _prices(inputs)
    values = full[params.price_field]
    visible = _windowed(inputs, full).index
    lines = {}
    warnings = []
    for period in params.periods:
        if not 2 <= period <= 400:
            raise InsufficientDataError("MA periods must be between 2 and 400")
        if len(values) < period:
            warnings.append(f"{params.kind.upper()}({period}) needs {period} sessions")
            continue
        lines[f"{params.kind}_{period}"] = _clean(
            moving_average(values, params.kind, period).loc[visible]
        )
    return AnalysisOutput("series", {"dates": _dates(visible), "lines": lines}, tuple(warnings))


# ── T-012 crossover & distance ───────────────────────────────────────────────


class CrossoverParams(BaseModel):
    fast: int = Field(50, ge=2, le=400)
    slow: int = Field(200, ge=3, le=400)
    kind: MaKind = "sma"


@analysis(
    id="T-012",
    version="1.0.0",
    title="MA crossover & distance",
    description=(
        "Golden (fast crosses above slow) and death crosses over the full history, and the "
        "current distance of price to each MA in % and in σ of daily returns × √period."
    ),
    applies_to=PRICE_SUBJECTS,
    params=CrossoverParams,
    result_kind="table",
    requires={"prices"},
    min_obs=200,
)
def crossovers(inputs: AnalysisInputs, params: CrossoverParams) -> AnalysisOutput:
    if params.fast >= params.slow:
        raise InsufficientDataError("The fast period must be shorter than the slow period")
    close = _prices(inputs)["close"]
    _require(len(close), params.slow + 1, "Crossovers")
    fast = moving_average(close, params.kind, params.fast)
    slow = moving_average(close, params.kind, params.slow)
    above = (fast > slow).where(slow.notna())
    changes = above.astype(float).diff()
    events = [
        {
            "date": pd.Timestamp(day).date().isoformat(),
            "event": "golden_cross" if change > 0 else "death_cross",
            "close": float(close.loc[day]),
        }
        for day, change in changes.items()
        if change in (1.0, -1.0) and pd.notna(above.shift(1).loc[day])
    ]
    sigma = float(close.pct_change().dropna().iloc[-252:].std(ddof=1))
    last = float(close.iloc[-1])
    distance: dict[str, dict[str, float | None]] = {}
    for name, line, period in (("fast", fast, params.fast), ("slow", slow, params.slow)):
        level = float(line.iloc[-1])
        pct = last / level - 1
        scaled = pct / (sigma * math.sqrt(period)) if sigma else None
        distance[name] = {"period": period, "ma": level, "pct": pct, "sigma": scaled}
    return AnalysisOutput(
        "table",
        {"events": events[-50:], "state": "above" if bool(above.iloc[-1]) else "below",
         "distance": distance},
    )  # fmt: skip


# ── T-013 volatility bands ───────────────────────────────────────────────────


class BandsParams(BaseModel):
    kind: Literal["bollinger", "keltner", "donchian"] = "bollinger"
    period: int = Field(20, ge=2, le=400)
    k: float = Field(2.0, gt=0, le=5)


def average_true_range(frame: pd.DataFrame, period: int) -> pd.Series:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous).abs(),
         (frame["low"] - previous).abs()], axis=1,
    ).max(axis=1)  # fmt: skip
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


@analysis(
    id="T-013",
    version="1.0.0",
    title="Volatility bands",
    description=(
        "Bollinger (SMA ± kσ, population σ), Keltner (EMA ± k·ATR, Wilder), Donchian "
        "(rolling high/low); %B = (P − lower)/(upper − lower), bandwidth = (upper − lower)/mid."
    ),
    applies_to=PRICE_SUBJECTS,
    params=BandsParams,
    result_kind="series",
    requires={"prices"},
    min_obs=2,
)
def bands(inputs: AnalysisInputs, params: BandsParams) -> AnalysisOutput:
    full = _prices(inputs)
    close = full["close"]
    _require(len(close), params.period, "Bands")
    if params.kind == "bollinger":
        mid = close.rolling(params.period).mean()
        spread = params.k * close.rolling(params.period).std(ddof=0)
        upper, lower = mid + spread, mid - spread
    elif params.kind == "keltner":
        mid = close.ewm(span=params.period, adjust=False, min_periods=params.period).mean()
        atr = average_true_range(full, params.period)
        upper, lower = mid + params.k * atr, mid - params.k * atr
    else:
        upper = full["high"].rolling(params.period).max()
        lower = full["low"].rolling(params.period).min()
        mid = (upper + lower) / 2
    width = upper - lower
    percent_b = (close - lower) / width.replace(0, np.nan)
    visible = _windowed(inputs, full).index
    return AnalysisOutput(
        "series",
        {
            "dates": _dates(visible),
            "upper": _clean(upper.loc[visible]),
            "mid": _clean(mid.loc[visible]),
            "lower": _clean(lower.loc[visible]),
            "percent_b": _clean(percent_b.loc[visible]),
            "bandwidth": _clean((width / mid).loc[visible]),
        },
    )


# ── T-030 52-week range ──────────────────────────────────────────────────────


@analysis(
    id="T-030",
    version="1.0.0",
    title="52-week range & distance",
    description=(
        "Position of the last close in the trailing 52-week high-low range, % from the high "
        "and the low, and drawdown from the all-time (stored history) high."
    ),
    applies_to=PRICE_SUBJECTS,
    params=NoParams,
    result_kind="table",
    requires={"prices"},
    min_obs=2,
)
def range_52w(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    full = _prices(inputs)
    last_day = pd.Timestamp(full.index[-1])
    year = full.loc[full.index > last_day - pd.Timedelta(days=365)]
    _require(len(year), 2, "52-week range")
    close = float(full["close"].iloc[-1])
    high, low = float(year["high"].max()), float(year["low"].min())
    peak = float(full["close"].max())
    warnings: tuple[str, ...] = ()
    if pd.Timestamp(full.index[0]) > last_day - pd.Timedelta(days=360):
        warnings = ("Less than one year of history: the range is partial.",)
    return AnalysisOutput(
        "table",
        {
            "close": close,
            "high_52w": high,
            "low_52w": low,
            "position": (close - low) / (high - low) if high > low else None,
            "from_high": close / high - 1,
            "from_low": close / low - 1,
            "ath": peak,
            "ath_date": pd.Timestamp(full["close"].idxmax()).date().isoformat(),
            "drawdown_from_ath": close / peak - 1,
        },
        warnings,
    )
