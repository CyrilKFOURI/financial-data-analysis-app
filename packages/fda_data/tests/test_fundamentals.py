"""Fundamentals: SEC/yfinance normalisation on recorded filings, PIT, ingestion, statements."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pandas as pd
import pytest
from sqlalchemy import Engine, insert, select

from fda_core.analyses.statements import (
    MultiYearParams,
    PeriodParams,
    StatementParams,
    YearsParams,
    balance_structure,
    build_periods,
    cash_bridge,
    income_sankey,
    multi_year,
    profit_waterfall,
    statement_quality,
)
from fda_core.registry import REGISTRY, AnalysisInputs, NotApplicableError
from fda_data.fundamentals import (
    SHARE_ITEMS,
    SecProvider,
    fiscal_period,
    ingest_fundamentals,
    load_facts,
    needs_share_supplement,
    normalize_sec,
    normalize_yfinance,
    supplement_shares,
)
from fda_data.providers.http import make_client
from fda_data.schema import fundamental_facts, instruments

FIXTURES = Path(__file__).parent / "fixtures"
AAPL = json.loads((FIXTURES / "sec" / "AAPL_companyfacts.json").read_text(encoding="utf-8"))


def mc_statements(_: str = "MC.PA") -> dict[str, pd.DataFrame]:
    raw = json.loads((FIXTURES / "yahoo" / "MC.PA_statements.json").read_text(encoding="utf-8"))
    return {
        name: pd.DataFrame(columns).rename(columns=pd.Timestamp) for name, columns in raw.items()
    }


def fy(facts: pd.DataFrame, code: str, year: int) -> float:
    rows = facts[
        (facts["line_item"] == code)
        & (facts["period_type"] == "FY")
        & (facts["fiscal_year"] == year)
    ]
    return float(rows.sort_values("filed_at")["value"].iloc[-1])


# ── normalisation ────────────────────────────────────────────────────────────


def test_fiscal_period_for_apple_and_calendar_years() -> None:
    assert fiscal_period(date(2025, 12, 27), 9) == (2026, 1)
    assert fiscal_period(date(2026, 6, 27), 9) == (2026, 3)
    assert fiscal_period(date(2023, 9, 30), 9) == (2023, 4)
    assert fiscal_period(date(2023, 10, 1), 9) == (2023, 4)  # 53-week year ending on Oct 1
    assert fiscal_period(date(2024, 3, 31), 12) == (2024, 1)


def test_sec_normalisation_matches_the_10k() -> None:
    facts = normalize_sec(AAPL)
    # Apple FY2024 10-K (filed 2024-11-01): revenue 391.035 bn, net income 93.736 bn.
    assert fy(facts, "revenue", 2024) == pytest.approx(391_035e6)
    assert fy(facts, "net_income", 2024) == pytest.approx(93_736e6)
    assert fy(facts, "total_assets", 2024) == pytest.approx(364_980e6)
    assert fy(facts, "capex", 2024) > 0  # canonical: outflows positive
    quarters = facts[(facts["period_type"] == "Q") & (facts["line_item"] == "revenue")]
    assert set(quarters["fiscal_quarter"].dropna().astype(int)) <= {1, 2, 3, 4}
    assert "H" not in set(facts["period_type"])  # US 10-Q YTD halves are skipped
    assert facts["filed_at"].notna().all()
    assert not facts["estimated_pit"].any()


def test_yfinance_fallback_flags_estimated_filing_dates() -> None:
    facts = normalize_yfinance(mc_statements(), "EUR")
    assert facts["estimated_pit"].all()
    revenue = facts[(facts["line_item"] == "revenue") & (facts["period_type"] == "FY")]
    assert len(revenue) >= 3
    assert (facts.loc[facts["line_item"] == "capex", "value"] >= 0).all()
    assert set(facts.loc[facts["line_item"] == "revenue", "unit"]) == {"EUR"}


# ── ingestion & point-in-time ────────────────────────────────────────────────


def sec_provider() -> SecProvider:
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    def handler(request: httpx.Request) -> httpx.Response:
        if "company_tickers" in str(request.url):
            return httpx.Response(200, json=tickers)
        if "CIK0000320193" in str(request.url):
            return httpx.Response(200, json=AAPL)
        return httpx.Response(404)

    return SecProvider(
        make_client(httpx.MockTransport(handler)), user_agent="test test@example.com"
    )


def add_instrument(
    engine: Engine, symbol: str, asset_class: str = "equity", currency: str = "USD"
) -> int:
    with engine.begin() as conn:
        return int(
            conn.execute(
                insert(instruments).values(
                    symbol=symbol,
                    name=symbol,
                    asset_class=asset_class,
                    currency=currency,
                    provider_symbols={},
                    profile={},
                )
            ).inserted_primary_key[0]
        )


def test_ingest_sec_then_point_in_time_view(engine: Engine) -> None:
    instrument_id = add_instrument(engine, "AAPL")
    result = ingest_fundamentals(engine, "AAPL", sec=sec_provider())
    assert result.status == "succeeded", result.message
    assert result.source == "sec_xbrl"
    latest = load_facts(engine, instrument_id)
    before_10k = load_facts(engine, instrument_id, as_of=date(2024, 10, 31))
    fy2024 = (
        (latest["line_item"] == "revenue")
        & (latest["period_type"] == "FY")
        & (latest["fiscal_year"] == 2024)
    )
    assert fy2024.sum() == 1
    known_before = (
        (before_10k["line_item"] == "revenue")
        & (before_10k["period_type"] == "FY")
        & (before_10k["fiscal_year"] == 2024)
    )
    assert known_before.sum() == 0  # the FY2024 10-K was filed on 2024-11-01
    again = ingest_fundamentals(engine, "AAPL", sec=sec_provider())  # replaces, never duplicates
    with engine.connect() as conn:
        count = len(conn.execute(select(fundamental_facts.c.id)).all())
    assert count == again.rows_written


SHARE_CONCEPTS = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ("us-gaap", "EarningsPerShareDiluted"),
    ("us-gaap", "EarningsPerShareBasic"),
)


def per_class_only() -> dict[str, Any]:
    """Apple's companyfacts as a Visa-like filer: share counts and EPS tagged only per class,
    so companyfacts has none of them."""
    facts = json.loads(json.dumps(AAPL))
    for taxonomy, concept in SHARE_CONCEPTS:
        facts["facts"].get(taxonomy, {}).pop(concept, None)
    return facts


def test_share_supplement_only_for_filers_without_current_counts() -> None:
    assert not needs_share_supplement(normalize_sec(AAPL))
    sec = normalize_sec(per_class_only())
    assert needs_share_supplement(sec)
    yahoo = normalize_yfinance(mc_statements(), "USD")
    merged = supplement_shares(sec, yahoo)
    added = merged.iloc[len(sec) :]
    assert len(added) > 0
    assert set(added["line_item"]) <= set(SHARE_ITEMS)
    assert set(added["source"]) == {"yfinance"}
    assert added["estimated_pit"].all()
    latest_income = pd.to_datetime(sec.loc[sec["line_item"] == "net_income", "period_end"]).max()
    assert (pd.to_datetime(added["period_end"]) <= latest_income).all()  # no flow-less period
    assert merged.iloc[: len(sec)].equals(sec)  # SEC facts untouched
    assert len(supplement_shares(merged, yahoo)) == len(merged)  # idempotent on its keys


def test_ingest_supplements_share_counts_for_per_class_filers(engine: Engine) -> None:
    instrument_id = add_instrument(engine, "AAPL")
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

    def handler(request: httpx.Request) -> httpx.Response:
        if "company_tickers" in str(request.url):
            return httpx.Response(200, json=tickers)
        return httpx.Response(200, json=per_class_only())

    sec = SecProvider(make_client(httpx.MockTransport(handler)), user_agent="t t@example.com")
    result = ingest_fundamentals(engine, "AAPL", sec=sec, yahoo_statements=mc_statements)
    assert (result.status, result.source) == ("succeeded", "sec_xbrl")
    stored = load_facts(engine, instrument_id)
    shares = stored[stored["line_item"].isin(SHARE_ITEMS)]
    assert len(shares) > 0
    assert set(shares["source"]) == {"yfinance"}

    def broken(_: str) -> dict[str, pd.DataFrame]:
        raise RuntimeError("yfinance down")

    again = ingest_fundamentals(engine, "AAPL", sec=sec, yahoo_statements=broken)
    assert again.status == "succeeded"  # SEC facts are kept when the supplement fails


def test_non_us_falls_back_to_yfinance_and_non_equity_is_refused(engine: Engine) -> None:
    add_instrument(engine, "MC.PA", currency="EUR")
    result = ingest_fundamentals(
        engine, "MC.PA", sec=sec_provider(), yahoo_statements=mc_statements
    )
    assert (result.status, result.source) == ("succeeded", "yfinance")
    add_instrument(engine, "BTC-USD", asset_class="crypto")
    assert ingest_fundamentals(engine, "BTC-USD", sec=sec_provider()).status == "failed"


# ── statement analyses on Apple's real filings ───────────────────────────────


def apple_inputs(**conv: Any) -> AnalysisInputs:
    facts = (
        normalize_sec(AAPL)
        .sort_values("filed_at")
        .drop_duplicates(["line_item", "period_type", "period_end"], keep="last")
    )
    return AnalysisInputs(
        data={"fundamentals": facts}, conventions={"asset_class": "equity"} | conv
    )


def test_statement_analyses_are_registered() -> None:
    for analysis_id in (
        "F-001",
        "F-002",
        "F-003",
        "F-004",
        "F-005",
        "F-006",
        "F-007",
        "F-008",
        "F-021",
        "F-022",
    ):
        assert REGISTRY.get(analysis_id).requires == frozenset({"fundamentals"})


def test_income_statement_fy_ttm_and_derived_q4() -> None:
    inputs = apple_inputs()
    out = REGISTRY.get("F-001").fn(inputs, StatementParams(common_size=True)).payload
    revenue = next(r for r in out["rows"] if r["code"] == "revenue")
    assert out["periods"][-2:] == ["FY2024", "FY2025"] or "FY2024" in out["periods"]
    assert revenue["common_size"][-1] == pytest.approx(1.0)
    assert next(r for r in out["rows"] if r["code"] == "ebitda")["derived"] is True
    quarterly = build_periods(inputs.data["fundamentals"], "Q")
    fy2024 = fy(inputs.data["fundamentals"], "revenue", 2024)
    q = quarterly.loc["revenue", [c for c in quarterly.columns if c.startswith("FY2024 ")]]
    assert q.sum() == pytest.approx(fy2024, rel=1e-9)  # Q4 derived = FY − Q1…Q3
    ttm = REGISTRY.get("F-001").fn(inputs, StatementParams(basis="TTM")).payload
    assert ttm["periods"][0].startswith("TTM")


def test_balance_sheet_and_cash_flow_tables() -> None:
    inputs = apple_inputs()
    bs = REGISTRY.get("F-002").fn(inputs, StatementParams()).payload
    assert {"total_assets", "total_equity", "net_debt"} <= {r["code"] for r in bs["rows"]}
    cf = REGISTRY.get("F-003").fn(inputs, StatementParams()).payload
    fcf = next(r for r in cf["rows"] if r["code"] == "fcf")
    cfo = next(r for r in cf["rows"] if r["code"] == "cfo")
    capex = next(r for r in cf["rows"] if r["code"] == "capex")
    assert fcf["values"][-1] == pytest.approx(cfo["values"][-1] - capex["values"][-1])


def test_visual_statements_and_quality_checks() -> None:
    inputs = apple_inputs()
    sankey = income_sankey(inputs, PeriodParams()).payload
    assert {"Revenue", "Gross profit", "Net income"} <= {n["name"] for n in sankey["nodes"]}
    bars = profit_waterfall(inputs, PeriodParams()).payload["bars"]
    assert bars[-1]["label"] == "Net income"
    assert bars[2]["value"] == pytest.approx(bars[0]["value"] + bars[1]["value"])
    bridge = cash_bridge(inputs, PeriodParams()).payload["bars"]
    assert bridge[3]["label"] == "CFO"
    structure = balance_structure(inputs, YearsParams()).payload
    assert "equity" in structure["liabilities_equity"]
    chart = multi_year(inputs, MultiYearParams()).payload
    assert 0.2 < chart["margins"]["operating"][-1] < 0.4
    quality = statement_quality(inputs, YearsParams()).payload
    latest = quality["checks"][-1]
    assert latest["balance_ok"] is True
    assert latest["gross_profit_ok"] is True


def test_statements_refuse_non_companies() -> None:
    with pytest.raises(NotApplicableError):
        REGISTRY.get("F-001").fn(apple_inputs(asset_class="crypto"), StatementParams())
