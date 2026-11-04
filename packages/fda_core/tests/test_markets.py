"""Golden tests for M19 markets analyses on synthetic, hand-checkable series."""

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.markets import (
    US_CURVE,
    CrossAssetParams,
    CurveParams,
    NoParams,
    RotationParams,
    cross_asset,
    curve_shape,
    level_changes,
    macro_row,
    markets_overview,
    performance,
    quadrant,
    recessions,
    returns_quilt,
    sector_rotation,
    sign_regimes,
    yield_curves,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

DAYS = pd.bdate_range("2020-01-01", "2025-12-31")


def market(
    prices: dict[str, pd.Series] | None = None, series: dict[str, pd.Series] | None = None
) -> AnalysisInputs:
    return AnalysisInputs(
        {"market_prices": prices or {}, "market_series": series or {}}, {"range": "1Y"}
    )


def geometric(daily: float, index: pd.DatetimeIndex = DAYS) -> pd.Series:
    return pd.Series(100 * (1 + daily) ** np.arange(len(index)), index=index)


def test_performance_uses_calendar_anchors_and_52w_position() -> None:
    s = pd.Series(
        [100.0, 110.0, 121.0], index=pd.to_datetime(["2024-12-31", "2025-06-30", "2025-12-31"])
    )
    row = performance(s, ("1d", "ytd", "1y"))
    assert row["return_1d"] == pytest.approx(0.1)
    assert row["return_ytd"] == pytest.approx(0.21)  # anchor = last close of the prior year
    assert row["return_1y"] == pytest.approx(0.21)
    assert row["position_52w"] == pytest.approx(1.0)
    rates = pd.Series([4.0, 4.25], index=pd.to_datetime(["2025-12-30", "2025-12-31"]))
    assert level_changes(rates, ("1d",))["change_bp_1d"] == pytest.approx(25)


def test_overview_lists_missing_inputs_and_refuses_without_data() -> None:
    with pytest.raises(InsufficientDataError, match="markets/ingest"):
        markets_overview(market(), NoParams())
    out = markets_overview(market({"^GSPC": geometric(0.001)}), NoParams()).payload
    assert [r["symbol"] for r in out["rows"]] == ["^GSPC"]
    assert "^NDX" in out["missing"]
    assert "DGS10" in out["missing"]


def test_rotation_quadrants_follow_relative_strength() -> None:
    days = pd.bdate_range("2020-01-03", "2025-12-26")  # whole weeks (Fridays)
    bench = geometric(0.0005, days)
    t = np.arange(len(days))
    accelerating = pd.Series(bench.to_numpy() * np.exp(1e-7 * t**2), index=days)
    lagging = pd.Series(bench.to_numpy() * np.exp(-0.0004 * t), index=days)
    out = sector_rotation(
        market({"SPY": bench, "XLK": accelerating, "XLU": lagging}), RotationParams()
    ).payload
    by = {p["symbol"]: p for p in out["rotation"]}
    assert by["XLK"]["quadrant"] == "leading"
    assert by["XLU"]["rs_ratio"] < 100
    assert len(by["XLK"]["tail_ratio"]) == 10
    assert quadrant(101, 99) == "weakening"
    assert quadrant(99, 101) == "improving"
    assert out["rank_1m"][0]["symbol"] == "XLK"


def test_curve_shapes_and_snapshots() -> None:
    normal = {0.25: 1.0, 2: 2.0, 5: 2.8, 10: 3.5, 30: 4.0}
    inverted = {0.25: 5.0, 2: 4.6, 5: 4.2, 10: 4.0, 30: 4.1}
    humped = {0.25: 2.0, 2: 3.5, 5: 3.4, 10: 2.6, 30: 2.4}
    assert curve_shape(normal)["shape"] == "normal"
    assert curve_shape(inverted)["shape"] == "inverted"
    assert curve_shape(humped)["shape"] == "humped"
    assert curve_shape(normal)["slope_10y_3m"] == pytest.approx(2.5)
    assert curve_shape(normal)["curvature_2_5_10"] == pytest.approx(2 * 2.8 - 2.0 - 3.5)
    rng = np.random.default_rng(0)
    level = 3 + np.cumsum(rng.normal(0, 0.03, len(DAYS)))
    series = {c: pd.Series(level + 0.1 * y, index=DAYS) for c, y in US_CURVE.items()}
    out = yield_curves(market(series=series), CurveParams(as_of="2023-06-30")).payload
    assert out["as_of"] == "2023-06-30"
    ten = next(r for r in out["rows"] if r["years"] == 10)
    assert ten["as_of"] == pytest.approx(series["DGS10"].loc["2023-06-30"])
    assert out["shape"]["shape"] == "normal"  # +0.1 pp per year of maturity: slope 0.975
    assert out["pca"][0]["component"] == "level"
    assert out["pca"][0]["explained"] > 0.99  # pure parallel shifts


def test_recession_periods_and_sign_regimes() -> None:
    months = pd.date_range("2019-01-01", periods=8, freq="MS")
    usrec = pd.Series([0, 0, 1, 1, 0, 1, 1, 1], index=months)
    assert recessions(usrec) == [
        {"start": "2019-03-01", "end": "2019-04-01"},
        {"start": "2019-06-01", "end": "2019-08-01"},
    ]
    corr = pd.Series([0.2, 0.3, -0.1, -0.2, 0.4], index=pd.bdate_range("2024-01-01", periods=5))
    regimes = sign_regimes(corr)
    assert [r["regime"] for r in regimes] == ["positive", "negative", "positive"]
    smoothed = sign_regimes(corr, min_sessions=3)
    assert [(r["regime"], r["sessions"]) for r in smoothed] == [("positive", 5)]
    assert regimes[1]["start"] == "2024-01-03"
    assert regimes[1]["end"] == "2024-01-04"


def test_macro_transforms() -> None:
    months = pd.date_range("2024-01-01", periods=13, freq="MS")
    cpi = pd.Series(100 * 1.002 ** np.arange(13), index=months)
    row = macro_row("CPIAUCSL", cpi)
    assert row["yoy"] == pytest.approx(1.002**12 - 1)
    assert row["ann_3m"] == pytest.approx(1.002**12 - 1)
    payrolls = pd.Series([100.0, 250.0, 400.0, 460.0], index=months[:4])
    flow = macro_row("PAYEMS", payrolls)
    assert flow["change"] == 60
    assert flow["change_3m_avg"] == pytest.approx(120)


def test_quilt_ranks_calendar_years() -> None:
    a = geometric(0.001)
    b = geometric(-0.0005)
    out = returns_quilt(market({"SPY": a, "TLT": b}), NoParams()).payload
    grid = out["grid"]
    assert grid["columns"] == ["US equity", "Long Treasuries"]
    assert grid["rows"][0] == "2020"
    year = a.loc["2021"]
    prior = a.loc["2020"].iloc[-1]
    assert grid["values"][1][0] == pytest.approx(year.iloc[-1] / prior - 1)
    assert all(best == 0 for best in grid["best"])
    assert out["ranked"][0]["rank_1"] == "US equity"


def test_cross_asset_stock_bond_correlation_from_yield_changes() -> None:
    rng = np.random.default_rng(1)
    shocks = rng.normal(0, 0.01, len(DAYS))
    equity = pd.Series(100 * np.exp(np.cumsum(shocks)), index=DAYS)
    # Yields rise when stocks rise: bond returns (−Δy) negatively correlated with stocks.
    yields = pd.Series(3 + np.cumsum(5 * shocks + rng.normal(0, 0.01, len(DAYS))), index=DAYS)
    vix = pd.Series(20 - 100 * shocks, index=DAYS)
    out = cross_asset(
        market({"^GSPC": equity, "^VIX": vix}, {"DGS10": yields, "BAMLH0A0HYM2": yields}),
        CrossAssetParams(),
    ).payload
    assert out["stock_bond_now"] < -0.8
    assert out["regimes"][-1]["regime"] == "negative"
    assert out["risk_components"] == ["hy_oas", "vix"]
    assert "risk_aversion" in out["lines"]
