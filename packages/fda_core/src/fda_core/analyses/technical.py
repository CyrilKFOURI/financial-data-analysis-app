"""Technical analysis (ANALYTICS_CATALOG TA-001…TA-012; module M14).

Pure numpy/pandas implementations of the TA-Lib definitions (Wilder smoothing for RSI/ATR/ADX,
EMA α = 2/(n+1) elsewhere). Indicators are computed on the full history and returned for the
context range; higher-timeframe values only use the previous closed bar (TA-009). Everything
here is descriptive: no buy/sell aggregation (TA-011).
"""

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from fda_core.analyses.price import PRICE_SUBJECTS, _prices, _windowed
from fda_core.analyses.stats import benjamini_hochberg
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis


def _iso(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def _clean(values: Any) -> list[float | None]:
    return [
        None if v is None or not np.isfinite(v) else float(v)
        for v in np.asarray(values, dtype=float)
    ]


def _ohlcv(inputs: AnalysisInputs, minimum: int = 30) -> pd.DataFrame:
    frame = _prices(inputs)
    if len(frame) < minimum:
        raise InsufficientDataError(f"Needs ≥ {minimum} sessions, got {len(frame)}")
    out = frame[["open", "high", "low", "close", "volume"]].astype(float)
    if "adj_close" in frame:  # back-adjust OHLC (volume inversely) so splits don't break indicators
        ratio = (frame["adj_close"] / frame["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].mul(
            ratio, axis=0
        )
        out["volume"] = out["volume"] / ratio
    return out


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (α = 1/n), as in RSI, ATR and ADX."""
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(f: pd.DataFrame) -> pd.Series:
    prev = f["close"].shift(1)
    return pd.concat(
        [f["high"] - f["low"], (f["high"] - prev).abs(), (f["low"] - prev).abs()], axis=1
    ).max(axis=1)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain, loss = wilder(delta.clip(lower=0), n), wilder(-delta.clip(upper=0), n)
    return 100 * gain / (gain + loss).replace(0, np.nan)


def stochastic(f: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    low, high = f["low"].rolling(k).min(), f["high"].rolling(k).max()
    pct_k = 100 * (f["close"] - low) / (high - low).replace(0, np.nan)
    slow_k = pct_k.rolling(d).mean()
    return slow_k, slow_k.rolling(d).mean()


def adx(f: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up, down = f["high"].diff(), -f["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=f.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=f.index)
    atr = wilder(true_range(f), n)
    plus_di = 100 * wilder(plus_dm, n) / atr
    minus_di = 100 * wilder(minus_dm, n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di


def parabolic_sar(f: pd.DataFrame, step: float = 0.02, maximum: float = 0.2) -> pd.Series:
    high, low = f["high"].to_numpy(), f["low"].to_numpy()
    sar = np.empty(len(f))
    long, af, ep = True, step, high[0]
    sar[0] = low[0]
    for i in range(1, len(f)):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if long:
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar[i]:
                long, sar[i], ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, maximum)
        else:
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar[i]:
                long, sar[i], ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, maximum)
    return pd.Series(sar, index=f.index)


def supertrend(f: pd.DataFrame, n: int = 10, k: float = 3.0) -> tuple[pd.Series, pd.Series]:
    atr = wilder(true_range(f), n)
    mid = (f["high"] + f["low"]) / 2
    upper, lower = (mid + k * atr).to_numpy(copy=True), (mid - k * atr).to_numpy(copy=True)
    close = f["close"].to_numpy()
    line, direction = np.full(len(f), np.nan), np.ones(len(f))
    for i in range(1, len(f)):
        if not np.isfinite(upper[i - 1]):
            continue
        upper[i] = min(upper[i], upper[i - 1]) if close[i - 1] <= upper[i - 1] else upper[i]
        lower[i] = max(lower[i], lower[i - 1]) if close[i - 1] >= lower[i - 1] else lower[i]
        direction[i] = (
            1 if close[i] > upper[i - 1] else -1 if close[i] < lower[i - 1] else direction[i - 1]
        )
        line[i] = lower[i] if direction[i] > 0 else upper[i]
    return pd.Series(line, index=f.index), pd.Series(direction, index=f.index)


def _visible(inputs: AnalysisInputs, f: pd.DataFrame) -> pd.Index:
    return _windowed(inputs, _prices(inputs)).index.intersection(f.index)


# ── TA-001 momentum / TA-002 trend / TA-003 volatility / TA-004 volume ───────


class OscillatorParams(BaseModel):
    period: int = Field(14, ge=2, le=100)


@analysis(
    id="TA-001",
    version="1.0.0",
    title="Momentum oscillators",
    description=(
        "RSI (Wilder), StochRSI, slow Stochastic %K/%D (14, 3), Williams %R, CCI (typical price, "
        "mean deviation), ROC and momentum over n, Ultimate oscillator (7/14/28)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="series",
    requires={"prices"},
    min_obs=30,
)
def momentum(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    f = _ohlcv(inputs)
    n = params.period
    close = f["close"]
    r = rsi(close, n)
    stoch_rsi = (
        100
        * (r - r.rolling(n).min())
        / (r.rolling(n).max() - r.rolling(n).min()).replace(0, np.nan)
    )
    k, d = stochastic(f, n)
    williams = (
        -100
        * (f["high"].rolling(n).max() - close)
        / (f["high"].rolling(n).max() - f["low"].rolling(n).min())
    )
    tp = (f["high"] + f["low"] + close) / 3
    mad = tp.rolling(n).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    cci = (tp - tp.rolling(n).mean()) / (0.015 * mad)
    bp = close - pd.concat([f["low"], close.shift(1)], axis=1).min(axis=1)
    tr = true_range(f)
    avg = {w: bp.rolling(w).sum() / tr.rolling(w).sum() for w in (7, 14, 28)}
    ultimate = 100 * (4 * avg[7] + 2 * avg[14] + avg[28]) / 7
    visible = _visible(inputs, f)
    lines = {
        "rsi": r,
        "stoch_rsi": stoch_rsi,
        "stoch_k": k,
        "stoch_d": d,
        "williams_r": williams,
        "cci": cci,
        "roc": 100 * close.pct_change(n),
        "momentum": close.diff(n),
        "ultimate": ultimate,
    }
    return AnalysisOutput(
        "series",
        {"dates": _iso(visible), "lines": {k_: _clean(v.loc[visible]) for k_, v in lines.items()}},
    )


@analysis(
    id="TA-002",
    version="1.0.0",
    title="Trend indicators",
    description=(
        "MACD (12, 26, 9) with histogram, ADX/+DI/−DI (14), Aroon up/down (25), Parabolic SAR "
        "(0.02, 0.2), SuperTrend (10, 3), Ichimoku (9, 26, 52, spans shifted forward 26) and a "
        "linear-regression channel (±2σ of residuals) over the visible range."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="series",
    requires={"prices"},
    min_obs=60,
)
def trend_indicators(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    f = _ohlcv(inputs, 60)
    close = f["close"]
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    adx_, plus_di, minus_di = adx(f, params.period)
    aroon_up = 100 * f["high"].rolling(26).apply(lambda w: np.argmax(w), raw=True) / 25
    aroon_down = 100 * f["low"].rolling(26).apply(lambda w: np.argmin(w), raw=True) / 25
    st_line, st_dir = supertrend(f)
    tenkan = (f["high"].rolling(9).max() + f["low"].rolling(9).min()) / 2
    kijun = (f["high"].rolling(26).max() + f["low"].rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((f["high"].rolling(52).max() + f["low"].rolling(52).min()) / 2).shift(26)
    visible = _visible(inputs, f)
    y = close.loc[visible].to_numpy()
    t = np.arange(len(y))
    slope, intercept = np.polyfit(t, y, 1)
    fit = intercept + slope * t
    resid_sd = float(np.std(y - fit, ddof=2))
    lines = {
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": macd - signal,
        "adx": adx_,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "aroon_up": aroon_up,
        "aroon_down": aroon_down,
        "psar": parabolic_sar(f),
        "supertrend": st_line,
        "tenkan": tenkan,
        "kijun": kijun,
        "span_a": span_a,
        "span_b": span_b,
    }
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(visible),
            "lines": {k: _clean(v.loc[visible]) for k, v in lines.items()},
            "regression_channel": {
                "mid": _clean(fit),
                "upper": _clean(fit + 2 * resid_sd),
                "lower": _clean(fit - 2 * resid_sd),
                "slope_per_session": float(slope),
            },
            "supertrend_direction": int(st_dir.iloc[-1]),
        },
    )


@analysis(
    id="TA-003",
    version="1.0.0",
    title="Volatility (technical)",
    description=(
        "ATR (Wilder, n), NATR = ATR/close, Chaikin volatility (ROC of the EMA(10) of high − low "
        "over 10) and the historical volatility cone (min/quartiles/max of rolling close-to-close "
        "volatility for 10…252-session windows, current value)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="series",
    requires={"prices"},
    min_obs=60,
)
def technical_volatility(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    f = _ohlcv(inputs, 60)
    factor = int(inputs.conventions.get("annualisation") or 252)
    atr = wilder(true_range(f), params.period)
    spread_ema = ema(f["high"] - f["low"], 10)
    chaikin = 100 * spread_ema.pct_change(10)
    returns = f["close"].pct_change()
    cone = []
    for window in (10, 21, 63, 126, 252):
        vol = returns.rolling(window).std() * math.sqrt(factor)
        series = vol.dropna()
        if len(series) < 20:
            continue
        cone.append(
            {
                "window": window,
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "current": float(series.iloc[-1]),
            }
        )
    visible = _visible(inputs, f)
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(visible),
            "lines": {
                "atr": _clean(atr.loc[visible]),
                "natr": _clean((100 * atr / f["close"]).loc[visible]),
                "chaikin_volatility": _clean(chaikin.loc[visible]),
            },
            "cone": cone,
        },
    )


class VolumeParams(BaseModel):
    bins: int = Field(40, ge=10, le=200)
    anchor: str | None = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Anchored-VWAP start date"
    )


@analysis(
    id="TA-004",
    version="1.0.0",
    title="Volume indicators",
    description=(
        "OBV, accumulation/distribution line, Chaikin money flow (20), MFI (14), volume "
        "oscillator (EMA 5 − EMA 20)/EMA 20, anchored VWAP (from the range start or a chosen "
        "date) and the volume profile of the range: price × volume histogram, point of control "
        "and 70 % value area."
    ),
    applies_to=PRICE_SUBJECTS,
    params=VolumeParams,
    result_kind="series",
    requires={"prices"},
    min_obs=30,
)
def volume_indicators(inputs: AnalysisInputs, params: VolumeParams) -> AnalysisOutput:
    if not inputs.conventions.get("has_volume", True):
        raise InsufficientDataError("No meaningful volume for this asset type")
    f = _ohlcv(inputs)
    close, high, low, volume = f["close"], f["high"], f["low"], f["volume"]
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    ad = (mfm.fillna(0) * volume).cumsum()
    cmf = (mfm * volume).rolling(20).sum() / volume.rolling(20).sum()
    tp = (high + low + close) / 3
    flow = tp * volume
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    mfi = 100 * pos / (pos + neg).replace(0, np.nan)
    vol_osc = 100 * (ema(volume, 5) - ema(volume, 20)) / ema(volume, 20)
    visible = _visible(inputs, f)
    start = pd.Timestamp(params.anchor) if params.anchor else visible[0]
    seg = f.loc[f.index >= start]
    vwap = ((seg["high"] + seg["low"] + seg["close"]) / 3 * seg["volume"]).cumsum() / seg[
        "volume"
    ].cumsum()
    window = f.loc[visible]
    typical = ((window["high"] + window["low"] + window["close"]) / 3).to_numpy()
    hist, edges = np.histogram(typical, bins=params.bins, weights=window["volume"].to_numpy())
    poc = int(np.argmax(hist))
    order = np.argsort(hist)[::-1]
    covered, area = 0.0, []
    for i in order:
        area.append(int(i))
        covered += hist[i]
        if covered >= 0.7 * hist.sum():
            break
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(visible),
            "lines": {
                "obv": _clean(obv.loc[visible]),
                "ad_line": _clean(ad.loc[visible]),
                "cmf": _clean(cmf.loc[visible]),
                "mfi": _clean(mfi.loc[visible]),
                "volume_oscillator": _clean(vol_osc.loc[visible]),
                "vwap": _clean(vwap.reindex(visible)),
            },
            "profile": {
                "edges": edges.tolist(),
                "volume": hist.tolist(),
                "poc": float((edges[poc] + edges[poc + 1]) / 2),
                "value_area_low": float(edges[min(area)]),
                "value_area_high": float(edges[max(area) + 1]),
            },
        },
    )


# ── TA-005 candlestick patterns ──────────────────────────────────────────────


def candle_patterns(f: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean detectors for 16 classic candlestick patterns (TA-Lib-like body/shadow rules
    relative to the 10-session average range; trend context from a 5-session close change)."""
    o, h, low, c = f["open"], f["high"], f["low"], f["close"]
    body = (c - o).abs()
    rng = (h - low).replace(0, np.nan)
    avg_body = body.rolling(10).mean()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - low
    bull, bear = c > o, c < o
    down_trend, up_trend = c.shift(1) < c.shift(5), c.shift(1) > c.shift(5)
    small = body <= 0.3 * avg_body
    long_ = body >= 1.0 * avg_body
    p = {  # previous candle
        "o": o.shift(1),
        "c": c.shift(1),
        "body": body.shift(1),
        "bull": bull.shift(1, fill_value=False),
        "bear": bear.shift(1, fill_value=False),
    }
    hammer_shape = (lower >= 2 * body) & (upper <= 0.25 * rng) & (body > 0)
    shooting_shape = (upper >= 2 * body) & (lower <= 0.25 * rng) & (body > 0)
    return {
        "doji": body <= 0.1 * rng,
        "hammer": hammer_shape & down_trend,
        "hanging_man": hammer_shape & up_trend,
        "inverted_hammer": shooting_shape & down_trend,
        "shooting_star": shooting_shape & up_trend,
        "bullish_engulfing": bull & p["bear"] & (o <= p["c"]) & (c >= p["o"]) & (body > p["body"]),
        "bearish_engulfing": bear & p["bull"] & (o >= p["c"]) & (c <= p["o"]) & (body > p["body"]),
        "bullish_harami": bull & p["bear"] & (p["body"] >= avg_body) & (o > p["c"]) & (c < p["o"]),
        "bearish_harami": bear & p["bull"] & (p["body"] >= avg_body) & (o < p["c"]) & (c > p["o"]),
        "piercing": bull & p["bear"] & (o < p["c"]) & (c > (p["o"] + p["c"]) / 2) & (c < p["o"]),
        "dark_cloud_cover": bear
        & p["bull"]
        & (o > p["c"])
        & (c < (p["o"] + p["c"]) / 2)
        & (c > p["o"]),
        "morning_star": bull
        & long_
        & small.shift(1, fill_value=False)
        & bear.shift(2, fill_value=False)
        & (c > (o.shift(2) + c.shift(2)) / 2),
        "evening_star": bear
        & long_
        & small.shift(1, fill_value=False)
        & bull.shift(2, fill_value=False)
        & (c < (o.shift(2) + c.shift(2)) / 2),
        "three_white_soldiers": bull
        & bull.shift(1, fill_value=False)
        & bull.shift(2, fill_value=False)
        & (c > c.shift(1))
        & (c.shift(1) > c.shift(2))
        & long_
        & (upper <= 0.3 * body),
        "three_black_crows": bear
        & bear.shift(1, fill_value=False)
        & bear.shift(2, fill_value=False)
        & (c < c.shift(1))
        & (c.shift(1) < c.shift(2))
        & long_
        & (lower <= 0.3 * body),
        "marubozu": (body >= 0.95 * rng) & long_,
    }


class PatternParams(BaseModel):
    horizon: int = Field(5, ge=1, le=60, description="Forward return horizon (sessions)")
    fdr: float = Field(0.10, gt=0, lt=0.5)


@analysis(
    id="TA-005",
    version="1.0.0",
    title="Candlestick patterns",
    description=(
        "Sixteen classic candlestick patterns with occurrence dates and, per pattern, the mean "
        "forward return over h sessions vs all sessions (Welch t-test), Benjamini–Hochberg "
        "adjusted at the chosen FDR — descriptive statistics, not signals."
    ),
    applies_to=PRICE_SUBJECTS,
    params=PatternParams,
    result_kind="table",
    requires={"prices"},
    min_obs=250,
)
def candlesticks(inputs: AnalysisInputs, params: PatternParams) -> AnalysisOutput:
    f = _ohlcv(inputs, 250)
    forward = f["close"].shift(-params.horizon) / f["close"] - 1
    base = forward.dropna().to_numpy()
    visible = _visible(inputs, f)
    rows, p_values = [], []
    for name, mask in candle_patterns(f).items():
        mask = mask.fillna(False).astype(bool)
        hits = forward[mask].dropna().to_numpy()
        row: dict[str, Any] = {
            "pattern": name,
            "occurrences": int(mask.sum()),
            "recent": _iso(f.index[mask & f.index.isin(visible)])[-5:],
        }
        if len(hits) >= 5:
            test = stats.ttest_ind(hits, base, equal_var=False)
            row |= {
                "mean_forward": float(hits.mean()),
                "baseline": float(base.mean()),
                "hit_rate": float((hits > 0).mean()),
                "p_value": float(test.pvalue),
            }
            p_values.append(float(test.pvalue))
        rows.append(row)
    tested = [r for r in rows if "p_value" in r]
    for row, (adj, sig) in zip(tested, benjamini_hochberg(p_values, params.fdr), strict=True):
        row["p_adjusted"], row["significant"] = adj, sig
    return AnalysisOutput(
        "table",
        {"horizon": params.horizon, "patterns": rows, "hypotheses": len(tested)},
        (
            "Historical forward statistics are descriptive; most patterns are not significant "
            "after FDR.",
        ),
    )


# ── TA-006 chart patterns (heuristic) ────────────────────────────────────────


def swings(close: pd.Series, order: int = 5) -> tuple[list[int], list[int]]:
    """Local maxima/minima indices over ±order sessions."""
    values = close.to_numpy()
    highs, lows = [], []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if values[i] == window.max():
            highs.append(i)
        if values[i] == window.min():
            lows.append(i)
    return highs, lows


@analysis(
    id="TA-006",
    version="1.0.0",
    title="Chart patterns (heuristic)",
    description=(
        "Swing-point heuristics on closes: double top/bottom (two extremes within 3 % separated "
        "by a ≥ 5 % retracement), head & shoulders (middle peak above two similar shoulders), "
        "ascending/descending/symmetric triangles (regressions of the last swing highs and lows). "
        "Labelled heuristic."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def chart_patterns(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    f = _windowed(inputs, _ohlcv(inputs, 60))
    close = f["close"]
    highs, lows = swings(close, max(params.period // 3, 3))
    c = close.to_numpy()
    found: list[dict[str, Any]] = []
    for kind, points, sign in (("double_top", highs, 1), ("double_bottom", lows, -1)):
        for a, b in itertools.pairwise(points):
            between = c[a : b + 1]
            retrace = (c[a] - between.min()) / c[a] if sign > 0 else (between.max() - c[a]) / c[a]
            if abs(c[a] - c[b]) / c[a] <= 0.03 and retrace >= 0.05:
                found.append(
                    {
                        "pattern": kind,
                        "start": _iso(close.index[[a]])[0],
                        "end": _iso(close.index[[b]])[0],
                        "level": float((c[a] + c[b]) / 2),
                    }
                )
    for l_, h_, r_ in zip(highs, highs[1:], highs[2:], strict=False):
        if c[h_] > c[l_] * 1.03 and c[h_] > c[r_] * 1.03 and abs(c[l_] - c[r_]) / c[l_] <= 0.05:
            found.append(
                {
                    "pattern": "head_and_shoulders",
                    "start": _iso(close.index[[l_]])[0],
                    "end": _iso(close.index[[r_]])[0],
                    "level": float(min(c[l_ : r_ + 1])),
                }
            )
    triangle = None
    if len(highs) >= 3 and len(lows) >= 3:
        hs, ls = highs[-4:], lows[-4:]
        slope_h = np.polyfit(hs, c[hs], 1)[0] / c[-1]
        slope_l = np.polyfit(ls, c[ls], 1)[0] / c[-1]
        flat = 0.0005
        if abs(slope_h) < flat and slope_l > flat:
            triangle = "ascending_triangle"
        elif slope_h < -flat and abs(slope_l) < flat:
            triangle = "descending_triangle"
        elif slope_h < -flat and slope_l > flat:
            triangle = "symmetric_triangle"
    return AnalysisOutput(
        "table",
        {
            "patterns": found[-20:],
            "current_triangle": triangle,
            "swing_highs": len(highs),
            "swing_lows": len(lows),
        },
        ("Heuristic detectors: shapes are approximate and not trading signals.",),
    )


# ── TA-007 levels ────────────────────────────────────────────────────────────


@analysis(
    id="TA-007",
    version="1.0.0",
    title="Levels",
    description=(
        "Support/resistance from clustering swing highs and lows (within 1.5 %), touch counts; "
        "classic, Fibonacci and Camarilla pivots from the last completed week; Fibonacci "
        "retracements and extensions of the largest swing in the range."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def levels(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    f = _windowed(inputs, _ohlcv(inputs, 60))
    close = f["close"]
    highs, lows = swings(close, max(params.period // 3, 3))
    points = sorted(close.iloc[highs + lows].to_numpy())
    clusters: list[list[float]] = []
    for value in points:
        if clusters and abs(value - np.mean(clusters[-1])) / np.mean(clusters[-1]) <= 0.015:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    last = float(close.iloc[-1])
    zones = sorted(
        (
            {
                "level": float(np.mean(cl)),
                "touches": len(cl),
                "kind": "resistance" if np.mean(cl) > last else "support",
            }
            for cl in clusters
            if len(cl) >= 2
        ),
        key=lambda z: abs(float(z["level"]) - last),
    )[:8]
    week = f.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).dropna().iloc[-2]
    hh, ll, cc = float(week["high"]), float(week["low"]), float(week["close"])
    pp = (hh + ll + cc) / 3
    span = hh - ll
    pivots = {
        "classic": {
            "pp": pp,
            "r1": 2 * pp - ll,
            "s1": 2 * pp - hh,
            "r2": pp + span,
            "s2": pp - span,
        },
        "fibonacci": {
            "pp": pp,
            "r1": pp + 0.382 * span,
            "r2": pp + 0.618 * span,
            "s1": pp - 0.382 * span,
            "s2": pp - 0.618 * span,
        },
        "camarilla": {
            "r3": cc + span * 1.1 / 4,
            "r4": cc + span * 1.1 / 2,
            "s3": cc - span * 1.1 / 4,
            "s4": cc - span * 1.1 / 2,
        },
    }
    lo_i, hi_i = int(np.argmin(close.to_numpy())), int(np.argmax(close.to_numpy()))
    lo_v, hi_v = float(close.iloc[lo_i]), float(close.iloc[hi_i])
    up = lo_i < hi_i  # retracement of the move from the earlier extreme to the later one
    move = hi_v - lo_v
    fib = {
        f"{r:.3f}": (hi_v - r * move if up else lo_v + r * move)
        for r in (0.236, 0.382, 0.5, 0.618, 0.786)
    }
    ext = {
        f"{r:.3f}": (hi_v + (r - 1) * move if up else lo_v - (r - 1) * move) for r in (1.272, 1.618)
    }
    return AnalysisOutput(
        "table",
        {
            "last": last,
            "zones": zones,
            "pivots": pivots,
            "fibonacci": {
                "swing_low": lo_v,
                "swing_high": hi_v,
                "direction": "up" if up else "down",
                "retracements": fib,
                "extensions": ext,
            },
        },
    )


# ── TA-008 relative strength / TA-009 multi-timeframe ────────────────────────


@analysis(
    id="TA-008",
    version="1.0.0",
    title="Relative strength",
    description=(
        "RS line = price / benchmark (rebased 100) and Mansfield RS = 100·(RS / SMA52w(RS) − 1) "
        "on weekly closes; RS change over 1/3/6/12 months."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="series",
    requires={"prices", "other_prices"},
    min_obs=60,
)
def relative_strength(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError(
            f"Ingest the benchmark {inputs.conventions.get('other_symbol')} for relative strength"
        )
    both = pd.concat([_prices(inputs)["close"], other["close"]], axis=1, join="inner").dropna()
    rs = both.iloc[:, 0] / both.iloc[:, 1]
    weekly = rs.resample("W-FRI").last().dropna()
    mansfield = 100 * (weekly / weekly.rolling(52).mean() - 1)
    visible = _windowed(inputs, _prices(inputs)).index
    rs_vis = rs.loc[rs.index.isin(visible)]
    changes = {f"{m}m": _change(rs, m) for m in (1, 3, 6, 12)}
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(rs_vis.index),
            "rs_line": _clean(100 * rs_vis / rs_vis.iloc[0]),
            "mansfield": {
                "dates": _iso(mansfield.dropna().index),
                "values": _clean(mansfield.dropna()),
            },
            "changes": changes,
            "benchmark": inputs.conventions.get("other_symbol"),
        },
    )


def _change(series: pd.Series, months: int) -> float | None:
    past = series.loc[series.index <= series.index[-1] - pd.DateOffset(months=months)]
    return float(series.iloc[-1] / past.iloc[-1] - 1) if len(past) else None


def _states(f: pd.DataFrame) -> dict[str, Any]:
    close = f["close"]
    r = rsi(close)
    macd = ema(close, 12) - ema(close, 26)
    sig = ema(macd, 9)
    adx_, plus_di, minus_di = adx(f)
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    mid, sd = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    pct_b = (close - (mid - 2 * sd)) / (4 * sd)
    k, _ = stochastic(f)

    def last(s: pd.Series) -> float | None:
        v = s.iloc[-1] if len(s) else np.nan
        return None if not np.isfinite(v) else float(v)

    def slope(s: pd.Series) -> str | None:
        v = s.diff(5).iloc[-1] if len(s) > 5 else np.nan
        return None if not np.isfinite(v) else "rising" if v > 0 else "falling"

    def zone(value: float | None, low: float, high: float) -> str | None:
        return (
            None
            if value is None
            else "high"
            if value >= high
            else "low"
            if value <= low
            else "neutral"
        )

    return {
        "rsi": {"value": last(r), "zone": zone(last(r), 30, 70), "slope": slope(r)},
        "stochastic_k": {"value": last(k), "zone": zone(last(k), 20, 80), "slope": slope(k)},
        "macd_hist": {
            "value": last(macd - sig),
            "zone": None
            if last(macd - sig) is None
            else "positive"
            if (last(macd - sig) or 0) > 0
            else "negative",
            "slope": slope(macd - sig),
        },
        "adx": {
            "value": last(adx_),
            "zone": zone(last(adx_), 20, 25),
            "slope": slope(adx_),
            "direction": None
            if last(plus_di) is None
            else "+DI"
            if (last(plus_di) or 0) > (last(minus_di) or 0)
            else "−DI",
        },
        "price_vs_sma50": {
            "value": last(close / sma50 - 1),
            "zone": None
            if last(sma50) is None
            else "above"
            if close.iloc[-1] > sma50.iloc[-1]
            else "below",
            "slope": slope(sma50),
        },
        "price_vs_sma200": {
            "value": last(close / sma200 - 1),
            "zone": None
            if last(sma200) is None
            else "above"
            if close.iloc[-1] > sma200.iloc[-1]
            else "below",
            "slope": slope(sma200),
        },
        "bollinger_pct_b": {
            "value": last(pct_b),
            "zone": zone(last(pct_b), 0, 1),
            "slope": slope(pct_b),
        },
    }


@analysis(
    id="TA-009",
    version="1.0.0",
    title="Multi-timeframe",
    description=(
        "The same indicator states on daily, weekly and monthly bars; weekly and monthly use only "
        "the previous closed bar (the current partial bar is dropped) — alignment table."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def multi_timeframe(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    f = _ohlcv(inputs, 60)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    frames = {"D": f}
    for name, rule in (("W", "W-FRI"), ("M", "ME")):
        bars = f.resample(rule).agg(agg).dropna()
        frames[name] = (
            bars.iloc[:-1] if len(bars) and bars.index[-1] >= f.index[-1] else bars
        )  # closed bars only
    table = {tf: _states(frame) for tf, frame in frames.items() if len(frame) >= 30}
    indicators = list(next(iter(table.values())).keys()) if table else []
    alignment = {ind: [table[tf][ind]["zone"] for tf in table] for ind in indicators}
    return AnalysisOutput(
        "table", {"timeframes": list(table), "states": table, "alignment": alignment}
    )


# ── TA-011 technical summary / TA-012 z-score ────────────────────────────────


@analysis(
    id="TA-011",
    version="1.0.0",
    title="Technical summary table",
    description=(
        "Current value, zone and 5-session slope of every core indicator on daily bars, in one "
        "dense table — no buy/sell aggregation."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OscillatorParams,
    result_kind="table",
    requires={"prices"},
    min_obs=30,
)
def technical_summary(inputs: AnalysisInputs, params: OscillatorParams) -> AnalysisOutput:
    states = _states(_ohlcv(inputs))
    return AnalysisOutput("table", {"rows": [{"indicator": k, **v} for k, v in states.items()]})


class ZParams(BaseModel):
    period: int = Field(50, ge=5, le=400)


@analysis(
    id="TA-012",
    version="1.0.0",
    title="Z-score & mean distance",
    description=(
        "z = (price − SMA_n) / std_n over the visible range, current percentile of |z| in the "
        "full history, and a regime hint from the rolling 252-session Hurst exponent of returns "
        "(> 0.55 trending, < 0.45 mean-reverting)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=ZParams,
    result_kind="series",
    requires={"prices"},
    min_obs=60,
)
def zscore_distance(inputs: AnalysisInputs, params: ZParams) -> AnalysisOutput:
    from fda_core.analyses.ts_memory import hurst_rs

    f = _ohlcv(inputs, max(60, params.period + 5))
    close = f["close"]
    z = (close - close.rolling(params.period).mean()) / close.rolling(params.period).std()
    valid = z.dropna()
    returns = close.pct_change().dropna().to_numpy()
    hurst = hurst_rs(returns[-252:])["anis_lloyd"] if len(returns) >= 252 else None
    regime = (
        None
        if hurst is None
        else "trending"
        if hurst > 0.55
        else "mean-reverting"
        if hurst < 0.45
        else "random-walk-like"
    )
    visible = _visible(inputs, f)
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(visible),
            "z": _clean(z.loc[visible]),
            "current": float(valid.iloc[-1]) if len(valid) else None,
            "abs_percentile": float((valid.abs() <= abs(valid.iloc[-1])).mean())
            if len(valid)
            else None,
            "hurst_252": hurst,
            "regime_hint": regime,
        },
    )
