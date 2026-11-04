"""Golden tests for CP-001 and CP-003 on synthetic members."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.comparison import (
    CompareParams,
    RankParams,
    comparison_set,
    member_kpis,
    peer_ranking,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError


def level(
    seed: int, n: int = 900, start: str = "2022-01-03", vol: float = 0.01, freq: str = "B"
) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, vol, n)), index=index)


def inp(*members: dict[str, object], range_: str = "MAX") -> AnalysisInputs:
    return AnalysisInputs(
        {"members": list(members)}, {"range": range_, "currency": "USD", "rf": 0.0}
    )


def m(label: str, series: pd.Series, **extra: object) -> dict[str, object]:
    return {"label": label, "series": series, "annualisation": 252, "kind": "asset"} | extra


def test_kpi_matrix_on_the_common_window_with_best_per_row() -> None:
    a = level(1)
    late = level(2, n=600, start="2022-07-01")
    bench = level(3)
    out = comparison_set(
        inp(m("A", a), m("LATE", late), m("BENCH", bench, is_benchmark=True)), CompareParams()
    ).payload
    assert out["window"]["start"] == late.index[0].date().isoformat()  # latest first date
    grid = out["grid"]
    rows = dict(zip(grid["rows"], grid["values"], strict=True))
    clipped = a.loc[(a.index >= late.index[0]) & (a.index <= late.index[-1])]  # common window
    assert rows["total_return"][0] == pytest.approx(clipped.iloc[-1] / clipped.iloc[0] - 1)
    vol = clipped.pct_change().dropna().std() * math.sqrt(252)
    assert rows["volatility"][0] == pytest.approx(vol)
    best = dict(zip(grid["rows"], grid["best"], strict=True))
    assert best["volatility"] == int(np.nanargmin(rows["volatility"]))
    assert rows["beta"][2] is None  # the benchmark is not compared with itself
    assert out["benchmark"] == "BENCH"


def test_scaled_copy_has_beta_one_and_zero_tracking_error() -> None:
    b = level(4)
    k = member_kpis(3 * b, 252, 0.0, b)
    assert k["beta"] == pytest.approx(1)
    assert k["tracking_error"] == pytest.approx(0, abs=1e-12)
    assert k["correlation"] == pytest.approx(1)


def test_overlays_calendar_correlation_stress_distribution() -> None:
    a, b = level(5), level(6)
    btc = level(7, n=1260, freq="D", vol=0.03)  # 7-day calendar
    members = (m("A", a), m("B", b), m("BTC", btc, annualisation=365))
    over = comparison_set(inp(*members), CompareParams(view="overlay")).payload
    assert all(v[0] == pytest.approx(100) for v in over["lines"].values())
    dd = comparison_set(inp(*members), CompareParams(view="drawdown")).payload
    assert all(max(x for x in v if x is not None) <= 1e-12 for v in dd["lines"].values())
    cal = comparison_set(inp(*members), CompareParams(view="calendar")).payload["grid"]
    year = cal["rows"].index("2023")
    y = a.loc["2023"]
    prev = a.loc[:"2022"].iloc[-1]
    assert cal["values"][year][0] == pytest.approx(y.iloc[-1] / prev - 1)
    corr = comparison_set(inp(*members), CompareParams(view="correlation")).payload
    assert np.allclose(np.diag(corr["matrix"]), 1)
    notes = comparison_set(inp(*members), CompareParams(view="kpi")).warnings
    assert any("annualisation" in n for n in notes)
    harmonised = comparison_set(inp(*members), CompareParams(harmonise=True)).payload
    assert set(harmonised["annualisation"].values()) == {252}
    q = comparison_set(inp(*members), CompareParams(view="distribution")).payload["grid"]
    assert q["rows"][0] == "p1"
    old = level(8, n=5000, start="2006-01-02")
    stress = comparison_set(inp(m("OLD", old), m("A", a)), CompareParams(view="stress")).payload[
        "grid"
    ]
    gfc = stress["rows"].index("GFC")
    assert stress["values"][gfc][1] is None  # A has no 2008 history
    with pytest.raises(InsufficientDataError):
        comparison_set(inp(m("A", a)), CompareParams())


def test_peer_ranking() -> None:
    members = [m(f"P{i}", level(10 + i), kind="portfolio") for i in range(5)]
    out = peer_ranking(inp(*members), RankParams(focus="P2")).payload
    rows = {r["metric"]: r for r in out["rows"]}
    assert rows["cagr"]["of"] == 5
    assert rows["cagr"]["percentile"] == pytest.approx((5 - rows["cagr"]["rank"]) / 4)
    assert peer_ranking(inp(*members), RankParams()).payload["focus"] == "P0"
