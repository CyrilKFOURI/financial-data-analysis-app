"""Golden tests for M20: screens, factor ranks, distributions, heatmap, breadth, price metrics."""

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.screener import (
    PRESETS,
    BreadthParams,
    MetricParams,
    RankParams,
    ScreenParams,
    breadth,
    distribution,
    factor_scores,
    heatmap,
    percentile_rank,
    price_metrics,
    run_screen,
    screen_diff,
    universe_breadth,
    winsorise,
    zscore,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError


def universe(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sectors = np.array(["Tech", "Energy", "Health"])[np.arange(n) % 3]
    frame = pd.DataFrame(
        {
            "symbol": [f"S{i:02d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "sector": sectors,
            "roic": np.where(sectors == "Tech", 0.25, 0.08) + rng.normal(0, 0.03, n),
            "fcf_yield": rng.uniform(0.0, 0.1, n),
            "net_cash": (np.arange(n) % 4 == 0).astype(float),
            "pe": rng.uniform(5, 50, n),
            "market_cap": rng.uniform(1e9, 1e12, n),
            "return_1m": rng.normal(0, 0.05, n),
        }
    )
    frame.loc[0:4, "roic"] = np.nan  # five members without the metric
    for metric in (
        "earnings_yield",
        "ev_ebit",
        "p_b",
        "gross_margin",
        "fcf_to_ni",
        "nd_ebitda",
        "momentum_12_1",
        "volatility_1y",
        "revenue_growth",
        "eps_growth",
        "revenue_cagr_3y",
        "dividend_yield",
    ):
        frame[metric] = rng.normal(0, 1, n)
    return frame


def test_screen_and_or_with_coverage_and_percentiles() -> None:
    frame = universe()
    tree = {
        "all": [
            {"metric": "roic", "op": ">", "value": 0.15},
            {
                "any": [
                    {"metric": "net_cash", "op": "=", "value": 1},
                    {"metric": "fcf_yield", "op": ">", "value": 0.04},
                ]
            },
        ]
    }
    out = run_screen(frame, ScreenParams(filters=tree, columns=["roic", "fcf_yield"]))
    known = frame.dropna(subset=["roic"])
    expected = known[
        (known["roic"] > 0.15) & ((known["net_cash"] == 1) | (known["fcf_yield"] > 0.04))
    ]
    assert out["count"] == len(expected)
    assert out["symbols"] == sorted(expected["symbol"])
    assert out["coverage"]["roic"] == {"with_data": 55, "excluded": 5}
    assert out["rows"][0]["roic"] == pytest.approx(expected["roic"].max())  # sorted by first column
    sector = run_screen(
        frame,
        ScreenParams(
            filters={
                "all": [{"metric": "pe", "op": "<=", "value": 30, "basis": "sector_percentile"}]
            }
        ),
    )
    ranks = percentile_rank(frame["pe"], frame["sector"])
    assert sector["count"] == int((ranks <= 30).sum())
    assert {r["sector"] for r in sector["sector_mix"]} == {"Tech", "Energy", "Health"}
    with pytest.raises(InsufficientDataError, match="Unknown metric"):
        run_screen(
            frame, ScreenParams(filters={"all": [{"metric": "nope", "op": ">", "value": 1}]})
        )
    assert screen_diff(["A", "B"], ["B", "C"]) == {"entrants": ["C"], "leavers": ["A"]}
    assert run_screen(frame, ScreenParams())["count"] == len(frame)  # empty tree keeps all
    assert set(PRESETS) >= {"quality_compounders", "deep_drawdown"}


def test_sector_neutral_composite_is_centred_within_each_sector() -> None:
    frame = universe(90)
    scores = factor_scores(frame, RankParams(sector_neutral=True))
    by_sector = scores["quality"].groupby(frame["sector"])
    assert by_sector.mean().abs().max() < 0.25  # ROIC level differences removed
    plain = factor_scores(frame, RankParams())
    assert plain["quality"].groupby(frame["sector"]).mean()["Tech"] > 0.2  # Tech has high ROIC
    assert "composite" in scores
    # Percentiles of the sector-neutral composite are spread evenly in every sector.
    pct = scores["composite"].groupby(frame["sector"]).rank(pct=True)
    for _, group in pct.groupby(frame["sector"]):
        assert group.min() < 0.1
        assert group.max() == 1.0


def test_winsorised_z_scores() -> None:
    values = pd.Series([*list(range(1, 100)), 10000.0])
    clipped = winsorise(values)
    assert clipped.max() < 10_000
    z = zscore(values)
    assert z.mean() == pytest.approx(0, abs=1e-12)
    assert z.std(ddof=0) == pytest.approx(1)


def test_distribution_heatmap_and_highlight() -> None:
    frame = universe()
    inputs = AnalysisInputs({"snapshot": frame}, {"as_of": "2026-10-23"})
    dist = distribution(inputs, MetricParams(metric="pe", highlight="S10", bins=10)).payload
    assert sum(b["count"] for b in dist["histogram"]) == len(frame)
    assert dist["highlight"]["value"] == pytest.approx(frame.loc[10, "pe"])
    assert {s["sector"] for s in dist["sectors"]} == {"Tech", "Energy", "Health"}
    heat = heatmap(inputs, MetricParams(metric="return_1m")).payload
    assert sum(len(g["children"]) for g in heat["tree"]) == len(frame)
    assert heat["centre"] == 0.0
    total = sum(g["value"] for g in heat["tree"])
    assert total == pytest.approx(frame["market_cap"].sum())


def test_price_metrics_on_a_known_path() -> None:
    days = pd.bdate_range("2023-01-02", "2025-12-31")
    close = pd.Series(100 * 1.001 ** np.arange(len(days)), index=days)
    frame = pd.DataFrame({"adj_close": close, "close": close, "volume": 1000.0})
    m = price_metrics(frame, None)
    one_year_ago = close.loc[: days[-1] - pd.Timedelta(days=365)].iloc[-1]
    assert m["return_12m"] == pytest.approx(close.iloc[-1] / one_year_ago - 1)
    assert m["momentum_12_1"] == pytest.approx(1.001**231 - 1)
    assert m["drawdown_from_high"] == 0
    assert m["position_52w"] == 1
    assert m["above_dma200"] == 1
    assert m["rsi_14"] == 100
    rng = np.random.default_rng(1)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
    levered = 50 * (bench / bench.iloc[0]) ** 1.5  # log returns = 1.5 × benchmark log returns
    noisy = pd.DataFrame({"adj_close": levered, "close": levered})
    assert price_metrics(noisy, bench)["beta_1y"] == pytest.approx(1.5)


def test_breadth_counts_and_mcclellan() -> None:
    days = pd.bdate_range("2024-01-01", periods=400)
    up = {f"U{i}": pd.Series(100 * 1.001 ** np.arange(400), index=days) for i in range(6)}
    down = {f"D{i}": pd.Series(100 * 0.999 ** np.arange(400), index=days) for i in range(4)}
    frame = breadth(up | down)
    last = frame.iloc[-1]
    assert last["pct_above_200dma"] == pytest.approx(60)
    assert last["advances"] == 6
    assert last["declines"] == 4
    assert frame["ad_line"].iloc[-1] == 2 * len(frame)
    assert last["new_highs"] == 6
    assert last["new_lows"] == 4
    assert last["mcclellan"] == pytest.approx(0, abs=1e-6)  # constant net advances
    out = universe_breadth(
        AnalysisInputs({"member_prices": up | down}, {}), BreadthParams(days=100)
    )
    assert len(out.payload["dates"]) == 100
    with pytest.raises(InsufficientDataError, match="≥ 5 members"):
        universe_breadth(
            AnalysisInputs({"member_prices": dict(list(up.items())[:3])}, {}), BreadthParams()
        )
