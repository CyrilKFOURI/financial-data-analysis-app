"""J6 ratios & scores: golden values on a hand-built company (computable by hand)."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.ratios import (
    AltmanParams,
    FyParams,
    ReturnsParams,
    ScorecardParams,
    altman,
    beneish,
    capital_allocation,
    dupont,
    efficiency,
    growth_metrics,
    health_scorecard,
    linear_score,
    liquidity,
    margins,
    montier,
    ohlson,
    piotroski,
    returns_on_capital,
    sector_set,
    solvency,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

# Three fiscal years of a simple industrial company (amounts in millions).
YEARS = {
    2023: dict(
        revenue=1000,
        cost_of_revenue=600,
        operating_income=150,
        d_and_a=50,
        interest_expense=10,
        pretax_income=140,
        income_tax=35,
        net_income=105,
        eps_diluted=1.05,
        shares_diluted_wavg=100,
        sga=200,
        cfo=160,
        capex=60,
        dividends_paid=40,
        buybacks=20,
        share_issuance=0,
        cash_and_equivalents=100,
        receivables=100,
        inventory=80,
        total_current_assets=300,
        ppe_net=500,
        total_assets=900,
        accounts_payable=60,
        total_current_liabilities=150,
        long_term_debt=250,
        total_liabilities=450,
        total_equity=450,
        retained_earnings=300,
        shares_outstanding=100,
    ),
    2024: dict(
        revenue=1100,
        cost_of_revenue=650,
        operating_income=176,
        d_and_a=55,
        interest_expense=10,
        pretax_income=166,
        income_tax=41.5,
        net_income=124.5,
        eps_diluted=1.245,
        shares_diluted_wavg=100,
        sga=210,
        cfo=180,
        capex=70,
        dividends_paid=44,
        buybacks=25,
        share_issuance=0,
        cash_and_equivalents=120,
        receivables=110,
        inventory=85,
        total_current_assets=330,
        ppe_net=520,
        total_assets=950,
        accounts_payable=65,
        total_current_liabilities=160,
        long_term_debt=240,
        total_liabilities=460,
        total_equity=490,
        retained_earnings=380,
        shares_outstanding=99,
    ),
    2025: dict(
        revenue=1210,
        cost_of_revenue=700,
        operating_income=200,
        d_and_a=60,
        interest_expense=9,
        pretax_income=191,
        income_tax=47.75,
        net_income=143.25,
        eps_diluted=1.45,
        shares_diluted_wavg=99,
        sga=230,
        cfo=210,
        capex=75,
        dividends_paid=50,
        buybacks=30,
        share_issuance=0,
        cash_and_equivalents=150,
        receivables=115,
        inventory=88,
        total_current_assets=365,
        ppe_net=540,
        total_assets=1000,
        accounts_payable=70,
        total_current_liabilities=165,
        long_term_debt=230,
        total_liabilities=470,
        total_equity=530,
        retained_earnings=470,
        shares_outstanding=98,
    ),
}


def facts() -> pd.DataFrame:
    rows = []
    for year, items in YEARS.items():
        for code, value in items.items():
            rows.append(
                {
                    "line_item": code,
                    "period_type": "FY",
                    "fiscal_year": year,
                    "fiscal_quarter": None,
                    "period_end": date(year, 12, 31),
                    "value": float(value) * (1 if code.startswith(("eps", "shares")) else 1e6),
                    "unit": "USD",
                    "filed_at": date(year + 1, 2, 15),
                    "source": "test",
                    "form": "10-K",
                    "is_restated": False,
                    "estimated_pit": False,
                }
            )
    return pd.DataFrame(rows)


def inputs(**conv: object) -> AnalysisInputs:
    return AnalysisInputs(
        data={"fundamentals": facts()},
        conventions={"asset_class": "equity", "sector": "Industrials", "country": "US"} | conv,
    )


def row(payload: dict, code: str) -> list[float | None]:
    return next(r["values"] for r in payload["rows"] if r["code"] == code)


def test_growth_and_margins() -> None:
    g = growth_metrics(inputs(), FyParams()).payload
    assert row(g, "revenue_yoy")[-1] == pytest.approx(0.10)
    assert g["summary"]["revenue"]["cagr_3y"] is None  # needs 4 years
    m = margins(inputs(), FyParams()).payload
    assert row(m, "gross_margin")[-1] == pytest.approx(510 / 1210)
    assert row(m, "ebit_margin")[-1] == pytest.approx(200 / 1210)
    assert row(m, "incremental_ebit_margin")[-1] == pytest.approx(24 / 110)


def test_returns_on_capital_golden() -> None:
    out = returns_on_capital(inputs(), ReturnsParams(wacc=0.08)).payload
    assert row(out, "roe")[-1] == pytest.approx(143.25 / ((490 + 530) / 2))
    assert row(out, "roa")[-1] == pytest.approx(143.25 / ((950 + 1000) / 2))
    tax = 47.75 / 191  # 25 %, inside the [0, 35 %] clip
    ic_now, ic_before = 230 + 530 - 150, 240 + 490 - 120
    assert row(out, "roic")[-1] == pytest.approx(200 * (1 - tax) / ((ic_now + ic_before) / 2))
    assert row(out, "roic_minus_wacc")[-1] == pytest.approx(row(out, "roic")[-1] - 0.08)


def test_dupont_product_equals_roe() -> None:
    d = dupont(inputs(), FyParams()).payload
    r = returns_on_capital(inputs(), ReturnsParams()).payload
    assert row(d, "roe_dupont")[-1] == pytest.approx(row(r, "roe")[-1])
    assert row(d, "tax_burden")[-1] * row(d, "interest_burden")[-1] * row(d, "ebit_margin")[
        -1
    ] == pytest.approx(row(d, "net_margin")[-1])


def test_liquidity_solvency_efficiency_golden() -> None:
    liq = liquidity(inputs(), FyParams()).payload
    assert row(liq, "current_ratio")[-1] == pytest.approx(365 / 165)
    assert row(liq, "quick_ratio")[-1] == pytest.approx((150 + 115) / 165)
    sol = solvency(inputs(), FyParams()).payload
    assert row(sol, "nd_ebitda")[-1] == pytest.approx((230 - 150) / 260)
    assert row(sol, "ebit_coverage")[-1] == pytest.approx(200 / 9)
    eff = efficiency(inputs(), FyParams()).payload
    assert row(eff, "dso")[-1] == pytest.approx((110 + 115) / 2 / 1210 * 365)
    assert row(eff, "dio")[-1] == pytest.approx((85 + 88) / 2 / 700 * 365)
    assert row(eff, "ccc")[-1] == pytest.approx(
        row(eff, "dso")[-1] + row(eff, "dio")[-1] - row(eff, "dpo")[-1]
    )


def test_capital_allocation_without_prices() -> None:
    out = capital_allocation(inputs(), FyParams()).payload
    assert row(out, "payout_ni")[-1] == pytest.approx(50 / 143.25)
    assert row(out, "shareholder_yield")[-1] is None  # no prices → no market cap
    assert out["dps_increase_streak"] == 2


def test_piotroski_by_hand() -> None:
    out = piotroski(inputs(), FyParams()).payload
    t = out["tests"]
    assert t["roa_positive"]
    assert t["cfo_positive"]
    assert t["accruals"]
    assert t["leverage_down"] is True  # LT debt 240 → 230 while assets grow
    assert t["no_dilution"] is True  # 99 → 98 shares
    assert out["score"] == sum(v for v in t.values() if v is not None)


def test_altman_z_double_prime_by_hand_and_financials_hidden() -> None:
    out = altman(inputs(country="FR"), AltmanParams()).payload
    assert out["variant"] == "z_double_prime"
    x1, x2, x3, x4 = (365 - 165) / 1000, 470 / 1000, 200 / 1000, 530 / 470
    assert out["latest"] == pytest.approx(6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4)
    assert out["zone"] == "safe"
    with pytest.raises(InsufficientDataError):
        altman(inputs(industry="Banks - Regional"), AltmanParams())


def test_beneish_and_montier_are_benign_for_steady_growth() -> None:
    b = beneish(inputs(), FyParams()).payload
    assert b["flag"] == "unlikely"
    assert b["variables"]["sgi"] == pytest.approx(1.1)
    c = montier(inputs(), FyParams()).payload
    assert c["score"] <= 3


def test_ohlson_probability_is_low_for_a_healthy_company() -> None:
    out = ohlson(inputs(), FyParams()).payload
    assert 0 < out["latest_probability"] < 0.2


def test_scorecard_blends_absolute_and_history() -> None:
    assert linear_score(0.15, 0.0, 0.25) == pytest.approx(60)
    assert linear_score(1.0, 4.0, 0.0) == pytest.approx(75)  # lower is better
    out = health_scorecard(inputs(), ScorecardParams()).payload
    assert out["threshold_set"] == "general"
    assert set(out["pillars"]) <= {"growth", "profitability", "balance_sheet", "cash_flow"}
    roic = next(d for d in out["drilldown"] if d["metric"] == "roic")
    assert roic["score"] == pytest.approx(0.7 * roic["absolute"] + 0.3 * roic["history_percentile"])
    assert 0 <= out["overall"] <= 100
    assert out["valuation"] == "needs stored prices"  # fundamentals only: no valuation pillar


def test_sector_sets() -> None:
    assert sector_set({"industry": "Banks - Diversified"}) == "banks"
    assert sector_set({"asset_subclass": "reit"}) == "reits"
    assert sector_set({"industry": "Software - Infrastructure"}) == "software"
    assert sector_set({"sector": "Utilities"}) == "utilities"
    assert sector_set({}) == "general"
    assert not math.isnan(np.float64(1.0))
