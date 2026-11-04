"""Spec 274: market capitalisation share count for multi-class issuers (Visa, Berkshire,
Alphabet), golden values from their mid-2026 figures."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.valuation import NoParams, current_multiples, resolve_shares
from fda_core.registry import AnalysisInputs, InsufficientDataError

QUARTER_ENDS = [
    date(2024, 9, 30),
    date(2024, 12, 31),
    date(2025, 3, 31),
    date(2025, 6, 30),
    date(2025, 9, 30),
    date(2025, 12, 31),
    date(2026, 3, 31),
    date(2026, 6, 30),
]


def fact(code: str, end: date, value: float, period_type: str = "Q") -> dict[str, object]:
    return {
        "line_item": code,
        "period_type": period_type,
        "fiscal_year": end.year,
        "fiscal_quarter": (end.month - 1) // 3 + 1 if period_type == "Q" else None,
        "period_start": None,
        "period_end": end,
        "value": value,
        "unit": "USD",
        "filed_at": end + timedelta(days=30),
        "source": "test",
        "form": "10-Q",
        "is_restated": False,
        "estimated_pit": False,
    }


def company(
    net_income: float,
    eps: float | None = None,
    outstanding: float | None = None,
    diluted: float | None = None,
    stale_outstanding: tuple[date, float] | None = None,
) -> pd.DataFrame:
    """Eight calendar quarters of flows; per-quarter EPS and share counts when given."""
    rows = []
    for end in QUARTER_ENDS:
        rows += [
            fact("revenue", end, net_income * 2),
            fact("operating_income", end, net_income * 1.3),
            fact("net_income", end, net_income),
            fact("cfo", end, net_income * 1.1),
            fact("capex", end, net_income * 0.1),
            fact("total_equity", end, net_income * 8),
        ]
        if eps is not None:
            rows.append(fact("eps_diluted", end, eps))
        if diluted is not None:
            rows.append(fact("shares_diluted_wavg", end, diluted))
    year_end = date(2025, 12, 31)
    rows += [
        fact("revenue", year_end, net_income * 8, "FY"),
        fact("operating_income", year_end, net_income * 5.2, "FY"),
        fact("net_income", year_end, net_income * 4, "FY"),
        fact("total_equity", year_end, net_income * 8, "FY"),
    ]
    if outstanding is not None:
        rows.append(fact("shares_outstanding", QUARTER_ENDS[-1], outstanding))
    if stale_outstanding is not None:
        rows.append(fact("shares_outstanding", *stale_outstanding))
    return pd.DataFrame(rows)


def prices(close: float) -> pd.DataFrame:
    index = pd.bdate_range("2025-10-01", "2026-10-23")
    return pd.DataFrame(
        {k: np.full(len(index), close) for k in ("open", "high", "low", "close", "adj_close")}
        | {"volume": 1e6},
        index=index,
    )


def multiples(facts: pd.DataFrame, close: float) -> dict[str, object]:
    out = current_multiples(
        AnalysisInputs(
            data={"fundamentals": facts, "prices": prices(close)},
            conventions={"asset_class": "equity", "currency": "USD"},
        ),
        NoParams(),
    )
    return dict(out.payload) | {"notes": out.warnings}


def test_visa_stale_pre_split_count_is_refused_not_used() -> None:
    # SEC companyfacts for V: only the 2009 cover-page class A count (pre 2015 4:1 split).
    facts = company(5.6e9, stale_outstanding=(date(2009, 12, 31), 470_210_301))
    with pytest.raises(InsufficientDataError, match="2009-12-31"):
        multiples(facts, 407.92)


def test_visa_as_converted_count_gives_the_real_pe() -> None:
    # yfinance supplement: 1.879 B as-converted outstanding, diluted w. avg overstated 2.106 B.
    facts = company(5.6e9, eps=2.96, outstanding=1_878_727_718, diluted=2_105_795_000)
    out = multiples(facts, 407.92)
    shares = out["shares"]
    assert shares["source"] == "shares_outstanding"
    assert shares["eps_implied"] == pytest.approx(22.4e9 / 11.84)
    assert out["market_cap"] == pytest.approx(407.92 * 1_878_727_718)
    assert out["multiples"]["pe"] == pytest.approx(34.2130, abs=1e-4)
    assert out["yields"]["fcf"] == pytest.approx(4 * 5.6e9 * 1.0 / (407.92 * 1_878_727_718))


def test_berkshire_class_a_equivalents_are_rejected_for_the_eps_consistent_count() -> None:
    # Outstanding is in class-A equivalents (1.43 M); diluted EPS is per class B share.
    facts = company(
        21.4e9, eps=21.4e9 / 2_154_663_423, outstanding=1_431_693, diluted=2_154_663_423
    )
    out = multiples(facts, 542.19)
    assert out["shares"]["source"] == "shares_diluted_wavg"
    assert out["multiples"]["pe"] == pytest.approx(542.19 * 2_154_663_423 / 85.6e9)
    assert any("diverge by more than 20 %" in n and "shares_outstanding" in n for n in out["notes"])


def test_alphabet_consistent_lines_need_no_note() -> None:
    facts = company(28e9, eps=28e9 / 12_309_000_000, outstanding=12_229_934_831)
    out = multiples(facts, 320.21)
    assert out["shares"]["source"] == "shares_outstanding"
    assert out["notes"] == ()


def test_eps_implied_count_when_every_line_diverges() -> None:
    facts = company(21.4e9, eps=21.4e9 / 2.15e9, outstanding=1_431_693)
    count = resolve_shares(facts)
    assert count.source == "shares_outstanding"  # no book given: no EPS cross-check
    out = multiples(facts, 542.19)
    assert out["shares"]["source"] == "eps_implied"
    assert out["shares"]["value"] == pytest.approx(2.15e9)


def test_uncrosschecked_count_is_flagged() -> None:
    out = multiples(company(5.6e9, outstanding=1.88e9), 400.0)
    assert out["shares"]["source"] == "shares_outstanding"
    assert any("not cross-checked" in n for n in out["notes"])
