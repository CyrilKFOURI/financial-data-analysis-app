"""Golden tests (hand-computed values) and oracle tests (empyrical-reloaded) for T-0xx."""

import math
from datetime import date

import empyrical
import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.price import (
    BandsParams,
    CrossoverParams,
    MovingAverageParams,
    NoParams,
    PriceChartParams,
    bands,
    base_100,
    cagr,
    compound_growth,
    crossovers,
    moving_average,
    moving_averages,
    period_returns,
    price_chart,
    range_52w,
    volatility,
    window_start,
)
from fda_core.registry import REGISTRY, AnalysisInputs, InsufficientDataError, NotApplicableError

EQUITY = {"annualisation": 252, "price_field": "adj_close", "has_volume": True}


def bars(
    close: list[float] | np.ndarray, start: str = "2024-01-02", freq: str = "B"
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    index = pd.date_range(start, periods=len(close), freq=freq)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(close), 1_000.0),
        },
        index=index,
    )


def gbm(n: int = 800, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return bars(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), start="2021-01-04")


def inputs(frame: pd.DataFrame, **conventions: object) -> AnalysisInputs:
    return AnalysisInputs(
        data={"prices": frame}, conventions=EQUITY | {"range": "MAX"} | conventions
    )


def test_all_price_analyses_are_registered_with_catalog_ids() -> None:
    for analysis_id in (
        "T-001",
        "T-002",
        "T-003",
        "T-005",
        "T-007",
        "T-011",
        "T-012",
        "T-013",
        "T-030",
    ):
        spec = REGISTRY.get(analysis_id)
        assert spec.requires == frozenset({"prices"})
        assert spec.applies_to == frozenset({"asset", "portfolio"})


# ── golden ───────────────────────────────────────────────────────────────────


def test_cagr_golden() -> None:
    # 100 → 121 over two years of 365.25 days = exactly 10 %/year.
    assert cagr(100, 121, 731) == pytest.approx(0.1, abs=2e-4)


def test_base_100_rebases_on_the_first_visible_close() -> None:
    out = base_100(inputs(bars([50, 55, 44])), NoParams())
    assert out.payload["values"] == pytest.approx([100, 110, 88])


def test_period_returns_golden() -> None:
    frame = bars(np.linspace(100, 200, 600), start="2023-01-02")
    rows = {r["period"]: r for r in period_returns(inputs(frame), NoParams()).payload["rows"]}
    assert rows["1D"]["return"] == pytest.approx(
        frame["adj_close"].iloc[-1] / frame["adj_close"].iloc[-2] - 1
    )
    assert rows["ITD"]["return"] == pytest.approx(1.0)
    assert rows["ITD"]["annualised"] is not None  # > 1 year
    assert rows["1M"]["annualised"] is None
    assert rows["5Y"]["return"] is None
    assert rows["5Y"]["reason"] == "history shorter than the period"


def test_ytd_is_measured_from_the_last_close_of_the_previous_year() -> None:
    frame = bars([100, 110, 121], start="2024-12-30", freq="D")  # 30 Dec, 31 Dec, 1 Jan
    rows = {r["period"]: r for r in period_returns(inputs(frame), NoParams()).payload["rows"]}
    assert rows["YTD"]["from"] == "2024-12-31"
    assert rows["YTD"]["return"] == pytest.approx(0.1)


def test_window_start() -> None:
    assert window_start("YTD", date(2026, 10, 22), date(2000, 1, 1)) == date(2026, 1, 1)
    assert window_start("MAX", date(2026, 10, 22), date(2000, 1, 1)) == date(2000, 1, 1)
    assert window_start("1Y", date(2026, 10, 22), date(2000, 1, 1)) == date(2025, 10, 22)


def test_moving_average_formulas() -> None:
    values = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert moving_average(values, "sma", 3).iloc[-1] == pytest.approx(5.0)
    assert moving_average(values, "wma", 3).iloc[-1] == pytest.approx((4 + 2 * 5 + 3 * 6) / 6)
    ema = moving_average(values, "ema", 3)
    alpha = 2 / (3 + 1)
    assert ema.iloc[3] == pytest.approx(alpha * 4 + (1 - alpha) * ema.iloc[2])
    # On a straight line DEMA/TEMA remove the EMA lag entirely once warmed up.
    line = pd.Series(np.arange(300, dtype=float))
    assert moving_average(line, "dema", 10).iloc[-1] == pytest.approx(299, abs=1e-6)
    assert moving_average(line, "tema", 10).iloc[-1] == pytest.approx(299, abs=1e-6)
    # HMA(16) with integer sub-periods keeps a 2/3-bar lag (WMA lags: 7/3, 15/3, then 1).
    assert moving_average(line, "hma", 16).iloc[-1] == pytest.approx(299 - 2 / 3, abs=1e-9)
    assert np.isfinite(moving_average(line, "kama", 10).iloc[-1])


def test_bollinger_golden() -> None:
    frame = bars([10, 12, 11, 13, 14])
    out = bands(inputs(frame), BandsParams(period=5, k=2)).payload
    window = np.array([10, 12, 11, 13, 14.0])
    assert out["mid"][-1] == pytest.approx(window.mean())
    assert out["upper"][-1] == pytest.approx(window.mean() + 2 * window.std(ddof=0))
    assert 0 < out["percent_b"][-1] < 1


def test_crossovers_detect_golden_and_death_crosses() -> None:
    up = np.concatenate(
        [np.linspace(100, 50, 250), np.linspace(50, 150, 250), np.linspace(150, 60, 250)]
    )
    out = crossovers(inputs(bars(up)), CrossoverParams(fast=20, slow=60)).payload
    kinds = [e["event"] for e in out["events"]]
    assert kinds == ["golden_cross", "death_cross"]
    assert out["state"] == "below"


def test_52_week_range_and_ath_drawdown() -> None:
    frame = bars([100, 150, 120])
    out = range_52w(inputs(frame), NoParams()).payload
    assert out["high_52w"] == pytest.approx(151.5)
    assert out["drawdown_from_ath"] == pytest.approx(120 / 150 - 1)
    assert out["ath_date"] == "2024-01-03"


def test_heikin_ashi_and_volume_hidden_for_fx() -> None:
    out = price_chart(
        inputs(bars([10, 11, 12]), has_volume=False), PriceChartParams(style="heikin_ashi")
    )
    assert out.payload["volume"] is None
    assert out.payload["close"][0] == pytest.approx((10 + 10.1 + 9.9 + 10) / 4)
    assert out.warnings


def test_moving_averages_keep_warm_up_outside_the_visible_range() -> None:
    frame = gbm()
    out = moving_averages(inputs(frame, range="1M"), MovingAverageParams(periods=[50])).payload
    assert all(v is not None for v in out["lines"]["sma_50"])


# ── guards ───────────────────────────────────────────────────────────────────


def test_small_samples_and_level_series_are_refused() -> None:
    with pytest.raises(InsufficientDataError):
        volatility(inputs(bars([1, 2, 3])), NoParams())
    with pytest.raises(NotApplicableError):
        volatility(inputs(gbm(), annualisation=None, price_field="level"), NoParams())
    with pytest.raises(InsufficientDataError):
        period_returns(AnalysisInputs(data={}, conventions=EQUITY), NoParams())


# ── oracle (empyrical-reloaded) ──────────────────────────────────────────────


@pytest.mark.parametrize("annualisation", [252, 365])
def test_volatility_matches_empyrical(annualisation: int) -> None:
    frame = gbm()
    ours = volatility(inputs(frame, annualisation=annualisation), NoParams()).payload["value"]
    returns = frame["adj_close"].pct_change().dropna()
    oracle = empyrical.annual_volatility(returns, annualization=annualisation)
    # empyrical uses ddof=1 too.
    assert ours == pytest.approx(oracle, rel=1e-10)


def test_cagr_matches_empyrical_on_calendar_days() -> None:
    frame = bars(100 * np.cumprod(np.full(366, 1.0004)), start="2024-01-01", freq="D")
    ours = compound_growth(inputs(frame), NoParams()).payload["value"]
    returns = frame["adj_close"].pct_change().dropna()
    oracle = empyrical.cagr(returns, annualization=365.25)
    assert ours == pytest.approx(oracle, rel=1e-3)
    assert ours == pytest.approx(1.0004**365.25 - 1, rel=1e-3)
    assert math.isfinite(ours)
