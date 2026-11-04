"""J8 peers, earnings & ownership on recorded MSFT provider data (offline)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, insert

from fda_core.analyses.peers import (
    EventParams,
    LiquidityParams,
    NoParams,
    analyst_views,
    benchmarking,
    comps,
    earnings_calendar,
    estimate_trends,
    event_study,
    holder_breakdown,
    insider_activity,
    institutional_changes,
    liquidity_profile,
    peer_group,
    peer_price_behaviour,
    sector_dashboard,
    short_interest,
    surprise_stats,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError
from fda_data.company import (
    Candidate,
    build_peer_group,
    edit_peer_group,
    ingest_company_data,
    load_peer_group,
    load_snapshots,
    score_candidates,
)
from fda_data.schema import instruments

FIXTURES = Path(__file__).parent / "fixtures" / "company"
PEERS = json.loads((FIXTURES / "msft_peers.json").read_text(encoding="utf-8"))
FRAMES = json.loads((FIXTURES / "msft_frames.json").read_text(encoding="utf-8"))


class RecordedYahoo:
    def info(self, symbol: str) -> dict[str, Any]:
        return dict(PEERS["infos"].get(symbol.upper(), {}))

    def industry_companies(self, industry_key: str) -> list[dict[str, Any]]:
        return list(PEERS["industry"])

    def sector_companies(self, sector_key: str) -> list[dict[str, Any]]:
        return []

    def weekly_closes(self, symbols: list[str]) -> pd.DataFrame:
        closes = PEERS["closes"]
        frame = pd.DataFrame(closes["columns"], index=pd.to_datetime(closes["index"]))
        return frame[[s for s in symbols if s in frame]].astype(float)

    def frames(self, symbol: str) -> dict[str, Any]:
        return {k: pd.DataFrame(v) if isinstance(v, list) else v for k, v in FRAMES.items()}


def add_company(engine: Engine, symbol: str = "MSFT") -> int:
    with engine.begin() as conn:
        return int(
            conn.execute(
                insert(instruments).values(
                    symbol=symbol,
                    name=symbol,
                    asset_class="equity",
                    currency="USD",
                    industry="Software - Infrastructure",
                    provider_symbols={},
                    profile={},
                )
            ).inserted_primary_key[0]
        )


def test_d_pe_2_scoring_respects_the_size_band_and_weights() -> None:
    subject = {
        "marketCap": 100.0,
        "industryKey": "x",
        "sectorKey": "s",
        "grossMargins": 0.6,
        "operatingMargins": 0.3,
    }
    closes = pd.DataFrame(
        {"S": np.linspace(1, 2, 60), "A": np.linspace(1, 2, 60), "B": np.linspace(2, 1, 60)}
    )
    cands = [
        Candidate(
            "A",
            "A",
            {"marketCap": 120.0, "industryKey": "x", "grossMargins": 0.6, "operatingMargins": 0.3},
        ),
        Candidate(
            "B",
            "B",
            {
                "marketCap": 90.0,
                "industryKey": "y",
                "sectorKey": "s",
                "grossMargins": 0.3,
                "operatingMargins": 0.1,
            },
        ),
        Candidate("C", "C", {"marketCap": 1000.0, "industryKey": "x"}),  # outside 0.25×–4×
    ]
    ranked = score_candidates(subject, cands, closes, "S")
    # Only two candidates inside 0.25×–4× (< 5): the band widens until enough peers exist.
    assert [c.symbol for c in ranked] == ["A", "B", "C"]
    assert ranked[0].breakdown["size_band"] == "any size"
    a = ranked[0]
    assert a.breakdown["industry"] == 1.0
    assert a.breakdown["margins"] == pytest.approx(1.0)
    assert a.score == pytest.approx(
        0.4 * 1 + 0.3 * a.breakdown["correlation"] + 0.2 * a.breakdown["size"] + 0.1 * 1
    )
    assert ranked[1].breakdown["industry"] == 0.5  # same sector only


def test_build_edit_and_freeze_peers(engine: Engine) -> None:
    instrument_id = add_company(engine)
    result = build_peer_group(engine, "MSFT", client=RecordedYahoo())
    assert result.status == "succeeded", result.message
    group = load_peer_group(engine, instrument_id)
    assert group["mode"] == "auto"
    assert 1 <= len(group["members"]) <= 10
    first = group["members"][0]["symbol"]
    edit_peer_group(engine, "MSFT", add=["ZZZZ"], remove=[first], pin=[])
    edited = load_peer_group(engine, instrument_id)
    assert edited["mode"] == "manual"
    assert first not in {m["symbol"] for m in edited["members"]}
    assert "ZZZZ" in {m["symbol"] for m in edited["members"]}
    kept = build_peer_group(engine, "MSFT", client=RecordedYahoo())
    assert "Manual peer group kept" in kept.message
    rebuilt = build_peer_group(engine, "MSFT", client=RecordedYahoo(), force=True)
    assert rebuilt.status == "succeeded"
    assert load_peer_group(engine, instrument_id)["mode"] == "auto"


def inputs(
    engine: Engine,
    instrument_id: int,
    prices: pd.DataFrame | None = None,
    bench: pd.DataFrame | None = None,
) -> AnalysisInputs:
    data: dict[str, Any] = {
        "peers": load_peer_group(engine, instrument_id),
        "company": load_snapshots(engine, instrument_id),
    }
    if prices is not None:
        data["prices"] = prices
    if bench is not None:
        data["other_prices"] = bench
    return AnalysisInputs(
        data=data,
        conventions={
            "asset_class": "equity",
            "price_field": "adj_close",
            "range": "MAX",
            "has_volume": True,
            "other_symbol": "SPY",
            "sector": "Technology",
        },
    )


def test_peer_analyses(engine: Engine) -> None:
    instrument_id = add_company(engine)
    build_peer_group(engine, "MSFT", client=RecordedYahoo())
    inp = inputs(engine, instrument_id)
    assert peer_group(inp, NoParams()).payload["members"]
    table = comps(inp, NoParams()).payload
    pe = next(r for r in table["rows"] if r["metric"] == "P/E")
    assert pe["subject"] == pytest.approx(PEERS["infos"]["MSFT"]["trailingPE"])
    assert pe["peer_median"] is not None
    assert (
        benchmarking(inp, NoParams()).payload["maps"]["roe_vs_growth"]["subject"]["x"] is not None
    )
    assert sector_dashboard(inp, NoParams()).payload["sector_etf"] == "XLK"
    behaviour = peer_price_behaviour(inp, NoParams()).payload
    assert -1 <= behaviour["correlation"] <= 1


def test_earnings_and_ownership(engine: Engine) -> None:
    instrument_id = add_company(engine)
    assert ingest_company_data(engine, "MSFT", client=RecordedYahoo()).status == "succeeded"
    inp = inputs(engine, instrument_id)
    calendar = earnings_calendar(inp, NoParams()).payload
    assert calendar["history"]
    assert calendar["source"] == "yfinance"
    stats = surprise_stats(inp, NoParams()).payload
    assert 0 <= stats["beat_rate"] <= 1
    assert estimate_trends(inp, NoParams()).payload["trend"]
    views = analyst_views(inp, NoParams())
    assert "not an opinion" in views.warnings[0]
    assert holder_breakdown(inp, NoParams()).payload["institutions_pct"] > 0.5
    assert institutional_changes(inp, NoParams()).payload["holders"]
    insiders = insider_activity(inp, NoParams()).payload
    assert set(insiders["net_value"]) == {"3m", "6m", "12m"}
    assert short_interest(inp, NoParams()).payload["shares_short"] > 0


def test_event_study_on_synthetic_reaction(engine: Engine) -> None:
    instrument_id = add_company(engine)
    ingest_company_data(engine, "MSFT", client=RecordedYahoo())
    inp = inputs(engine, instrument_id)
    dates = pd.to_datetime(
        pd.DataFrame(inp.data["company"]["earnings_dates"]["payload"])["Earnings Date"], utc=True
    )
    index = pd.bdate_range("2014-01-01", "2026-10-22")
    rng = np.random.default_rng(3)
    m = rng.normal(0.0004, 0.01, len(index))
    a = 1.1 * m + rng.normal(0, 0.008, len(index))
    for d in dates:
        local = d.tz_convert("America/New_York")
        day0 = pd.Timestamp(local.date()) + (
            pd.Timedelta(days=1) if local.hour >= 16 else pd.Timedelta(0)
        )
        i = index.searchsorted(day0)
        if i < len(index):
            a[i] += 0.04  # +4 % abnormal reaction on every report

    def frame(r):
        return pd.DataFrame(
            {k: 100 * np.cumprod(1 + r) for k in ("open", "high", "low", "close", "adj_close")}
            | {"volume": 1e6},
            index=index,
        )

    out = event_study(inputs(engine, instrument_id, frame(a), frame(m)), EventParams()).payload
    day0 = out["days"].index(0)
    jump = out["mean_car"][day0] - out["mean_car"][day0 - 1]
    assert jump == pytest.approx(0.04, abs=0.01)
    liq = liquidity_profile(inputs(engine, instrument_id, frame(a)), LiquidityParams()).payload
    assert liq["days_to_trade"] > 0


def test_missing_data_is_explained(engine: Engine) -> None:
    instrument_id = add_company(engine)
    with pytest.raises(InsufficientDataError, match="build peers"):
        comps(inputs(engine, instrument_id), NoParams())
    with pytest.raises(InsufficientDataError, match="ingest company data"):
        earnings_calendar(inputs(engine, instrument_id), NoParams())
