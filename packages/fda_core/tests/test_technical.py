"""Golden tests for TA-001…TA-012 (hand-computed values and no-look-ahead checks)."""

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.technical import (
    OscillatorParams,
    PatternParams,
    VolumeParams,
    ZParams,
    candlesticks,
    chart_patterns,
    levels,
    momentum,
    multi_timeframe,
    parabolic_sar,
    relative_strength,
    rsi,
    stochastic,
    technical_summary,
    technical_volatility,
    trend_indicators,
    true_range,
    volume_indicators,
    zscore_distance,
)
from fda_core.registry import REGISTRY, AnalysisInputs, InsufficientDataError

from .test_price import bars, gbm, inputs


def test_all_technical_ids_registered() -> None:
    ids = {f"TA-{i:03d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)}
    assert all(i in REGISTRY for i in ids)


def test_rsi_extremes_and_wilder_recursion() -> None:
    up = pd.Series(np.arange(1.0, 40.0))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    assert rsi(-up).iloc[-1] == pytest.approx(0.0)
    close = pd.Series(
        [10, 11, 10.5, 11.5, 11, 12, 11.8, 12.5, 12, 13, 12.6, 13.4, 13, 14, 13.5, 14.2, 13.9]
    )
    delta = close.diff().to_numpy()[1:]
    g, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    n, ag, al = 14, g[0], loss[0]
    for i in range(1, len(delta)):  # Wilder recursion seeded with the first change
        ag, al = ag + (g[i] - ag) / n, al + (loss[i] - al) / n
    assert rsi(close).iloc[-1] == pytest.approx(100 * ag / (ag + al))


def test_stochastic_true_range_and_psar() -> None:
    f = bars(np.linspace(100, 130, 40))
    k, d = stochastic(f)
    assert 85 < k.iloc[-1] <= 100
    assert 85 < d.iloc[-1] <= 100
    tr = true_range(f)
    assert tr.iloc[5] == pytest.approx(f["high"].iloc[5] - f["low"].iloc[5])
    gap = bars([100.0, 100.0, 80.0])  # a gap down: TR reaches back to the previous close
    assert true_range(gap).iloc[2] == pytest.approx(100.0 - 80 * 0.99)
    swing = bars(np.r_[np.linspace(100, 130, 30), np.linspace(129, 90, 30)])
    sar = parabolic_sar(swing)
    assert sar.iloc[20] < swing["low"].iloc[20]  # below price in the up-leg
    assert sar.iloc[-1] > swing["high"].iloc[-1]  # above price after the flip


def test_momentum_trend_volatility_shapes() -> None:
    inp = inputs(gbm())
    m = momentum(inp, OscillatorParams()).payload
    assert set(m["lines"]) >= {"rsi", "stoch_rsi", "cci", "ultimate", "williams_r"}
    assert len(m["dates"]) == len(m["lines"]["rsi"])
    assert all(v is None or 0 <= v <= 100 for v in m["lines"]["rsi"])
    assert all(v is None or -100 <= v <= 0 for v in m["lines"]["williams_r"])
    t = trend_indicators(inp, OscillatorParams()).payload
    hist = np.array(t["lines"]["macd_hist"][-50:], dtype=float)
    macd = np.array(t["lines"]["macd"][-50:], dtype=float)
    sig = np.array(t["lines"]["macd_signal"][-50:], dtype=float)
    assert np.allclose(hist, macd - sig)
    assert t["supertrend_direction"] in (-1, 1)
    v = technical_volatility(inp, OscillatorParams()).payload
    cone = {c["window"]: c for c in v["cone"]}
    assert cone[21]["min"] <= cone[21]["median"] <= cone[21]["max"]


def test_volume_indicators_golden() -> None:
    f = bars([10.0, 11, 10, 12, 12] * 8)
    f["volume"] = np.arange(1.0, 41.0)
    out = volume_indicators(inputs(f), VolumeParams(bins=10)).payload
    # OBV: +v on up days, −v on down days, 0 when unchanged.
    sign = np.nan_to_num(np.sign(np.diff(f["close"].to_numpy(), prepend=np.nan)))
    assert out["lines"]["obv"][-1] == pytest.approx(float(np.sum(sign * f["volume"].to_numpy())))
    profile = out["profile"]
    assert profile["value_area_low"] <= profile["poc"] <= profile["value_area_high"]
    assert profile["poc"] == pytest.approx(12.0, abs=0.3)  # most volume traded around 12
    with pytest.raises(InsufficientDataError):
        volume_indicators(inputs(f, has_volume=False), VolumeParams())


def test_split_does_not_break_indicators() -> None:
    f = gbm(300)
    split = f.copy()
    split.loc[split.index[:150], ["open", "high", "low", "close"]] *= 4  # raw prices pre 4:1 split
    a = momentum(inputs(f), OscillatorParams()).payload["lines"]["rsi"][-1]
    b = momentum(inputs(split), OscillatorParams()).payload["lines"]["rsi"][-1]
    assert a == pytest.approx(b)


def test_multi_timeframe_uses_closed_bars_only() -> None:
    f = gbm(800)
    f = f.loc[f.index <= f.index[f.index.day_of_week == 2][-1]]  # ends on a Wednesday
    base = multi_timeframe(inputs(f), OscillatorParams()).payload
    spiked = f.copy()
    spiked.iloc[-1, :4] *= 1.5  # move the partial week's bar
    spiked.iloc[-1, spiked.columns.get_loc("adj_close")] *= 1.5
    out = multi_timeframe(inputs(spiked), OscillatorParams()).payload
    assert base["timeframes"] == ["D", "W", "M"]
    assert out["states"]["W"] == base["states"]["W"]
    assert out["states"]["M"] == base["states"]["M"]
    assert out["states"]["D"]["rsi"]["value"] != base["states"]["D"]["rsi"]["value"]
    rows = technical_summary(inputs(f), OscillatorParams()).payload["rows"]
    assert {r["indicator"] for r in rows} >= {"rsi", "adx", "macd_hist"}
    assert all("buy" not in str(r).lower() and "sell" not in str(r).lower() for r in rows)


def test_candlestick_stats_are_fdr_adjusted() -> None:
    out = candlesticks(inputs(gbm(1000)), PatternParams()).payload
    tested = [r for r in out["patterns"] if "p_value" in r]
    assert out["hypotheses"] == len(tested) > 0
    assert all(r["p_adjusted"] >= r["p_value"] - 1e-12 for r in tested)
    assert len(out["patterns"]) == 16


def test_double_top_and_levels() -> None:
    path = np.r_[
        np.linspace(100, 120, 20),
        np.linspace(120, 105, 15),
        np.linspace(105, 120.5, 15),
        np.linspace(120.5, 100, 20),
    ]
    found = chart_patterns(inputs(bars(path)), OscillatorParams()).payload["patterns"]
    assert any(
        p["pattern"] == "double_top" and p["level"] == pytest.approx(120.25, abs=0.5) for p in found
    )
    f = gbm(300)
    out = levels(inputs(f), OscillatorParams()).payload
    week = f.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).iloc[-2]
    pp = (week["high"] + week["low"] + week["close"]) / 3
    assert out["pivots"]["classic"]["pp"] == pytest.approx(pp)
    assert out["pivots"]["classic"]["r1"] == pytest.approx(2 * pp - week["low"])
    fib = out["fibonacci"]
    lo, hi = sorted((fib["swing_low"], fib["swing_high"]))
    assert lo <= fib["retracements"]["0.500"] <= hi


def test_relative_strength_constant_ratio() -> None:
    bench = gbm(400)
    asset = bench.copy()
    asset[["open", "high", "low", "close", "adj_close"]] *= 2
    inp = AnalysisInputs(
        data={"prices": asset, "other_prices": bench},
        conventions={"range": "MAX", "price_field": "adj_close", "other_symbol": "SPY"},
    )
    out = relative_strength(inp, OscillatorParams()).payload
    assert np.allclose(np.array(out["rs_line"], dtype=float), 100)
    assert np.allclose(np.array(out["mansfield"]["values"], dtype=float), 0, atol=1e-9)
    assert out["changes"]["3m"] == pytest.approx(0)


def test_zscore_and_regime_hint() -> None:
    f = gbm(600)
    out = zscore_distance(inputs(f), ZParams(period=20)).payload
    c = f["close"]
    z = (c - c.rolling(20).mean()) / c.rolling(20).std()
    assert out["current"] == pytest.approx(z.iloc[-1])
    assert out["regime_hint"] in {"trending", "mean-reverting", "random-walk-like"}
    assert 0 < out["abs_percentile"] <= 1
