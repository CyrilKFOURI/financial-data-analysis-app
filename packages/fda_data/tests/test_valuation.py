"""J7 valuation on Apple's recorded filings with synthetic prices: formulas, PIT, round-trips."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.valuation import (
    DcfParams,
    DdmParams,
    HistoryParams,
    NoParams,
    SensitivityParams,
    SotpParams,
    WaccParams,
    current_multiples,
    dcf,
    dcf_assumptions,
    dividend_discount,
    earnings_power,
    football_field,
    implied_expectations,
    multiple_decomposition,
    pit_daily,
    pit_multiples,
    pit_ttm,
    residual_income,
    reverse_dcf,
    run_dcf,
    sensitivity,
    sum_of_parts,
    wacc_builder,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError
from fda_data.fundamentals import normalize_sec

FIXTURES = Path(__file__).parent / "fixtures"
ALL = normalize_sec(
    json.loads((FIXTURES / "sec" / "AAPL_companyfacts.json").read_text(encoding="utf-8"))
)
LATEST = ALL.sort_values("filed_at").drop_duplicates(
    ["line_item", "period_type", "period_end"], keep="last"
)


def prices(
    level: float = 200.0, start: str = "2022-01-03", end: str = "2026-10-22", seed: int = 1
) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = level * np.exp(
        np.cumsum(rng.normal(0, 0.012, len(index)))
        - np.cumsum(rng.normal(0, 0.012, len(index))).mean() * 0
    )
    close = close / close[-1] * level
    return pd.DataFrame(
        {k: close for k in ("open", "high", "low", "close", "adj_close")} | {"volume": 1e7},
        index=index,
    )


def inputs(**conv: object) -> AnalysisInputs:
    p = prices()
    return AnalysisInputs(
        data={
            "fundamentals": LATEST,
            "fundamentals_all": ALL,
            "prices": p,
            "other_prices": prices(500, seed=2),
        },
        conventions={
            "asset_class": "equity",
            "currency": "USD",
            "rf": 0.04,
            "rf_source": "test",
            "other_symbol": "SPY",
        }
        | conv,
    )


def test_current_multiples_bridge() -> None:
    out = current_multiples(inputs(), NoParams()).payload
    assert out["price"] == pytest.approx(200.0)
    assert out["enterprise_value"] == pytest.approx(out["market_cap"] + out["net_debt"], rel=0.05)
    assert 5 < out["multiples"]["pe"] < 60
    assert out["yields"]["earnings"] == pytest.approx(1 / out["multiples"]["pe"])


def test_wacc_is_consistent() -> None:
    w = wacc_builder(inputs(), WaccParams(beta=1.2)).payload
    assert w["cost_of_equity"] == pytest.approx(0.04 + 1.2 * 0.05)
    assert w["wacc"] == pytest.approx(
        w["equity_weight"] * w["cost_of_equity"]
        + w["debt_weight"] * w["cost_of_debt"] * (1 - w["tax_rate"])
    )
    assert w["default_spread"] <= 0.0107  # Apple's coverage is very high


def test_dcf_hand_check_on_flat_firm() -> None:
    """Zero growth, g = 0: value = NOPAT / WACC (perpetuity), up to mid-year timing."""
    params = DcfParams(growth=0.0, terminal_growth=0.0, wacc_override=0.08, margin=0.30)
    a = dcf_assumptions(inputs(), params)
    out = run_dcf(a, params)
    nopat = a["revenue0"] * 0.30 * (1 - a["tax"])
    expected_ev = nopat / 0.08 * (1.08**0.5)  # mid-year convention shifts cash flows half a year
    assert out["enterprise_value"] == pytest.approx(expected_ev, rel=1e-6)


def test_reverse_dcf_round_trip() -> None:
    out = reverse_dcf(inputs(), DcfParams(beta=1.0)).payload
    assert out["implied_growth"] is not None
    a = dcf_assumptions(inputs(), DcfParams(beta=1.0))
    value = run_dcf(a, DcfParams(beta=1.0), growth=out["implied_growth"])["per_share"]
    assert value == pytest.approx(out["price"], rel=1e-4)


def test_dcf_and_sensitivity_payloads() -> None:
    d = dcf(inputs(), DcfParams(beta=1.0)).payload
    assert len(d["rows"]) == 10
    assert d["per_share"] > 0
    s = sensitivity(inputs(), SensitivityParams(beta=1.0, n_sims=300)).payload
    base = s["scenarios"]["base"]
    assert s["scenarios"]["bear"] < base < s["scenarios"]["bull"]
    centre = s["wacc_g"]["values"][2][2]
    assert centre == pytest.approx(base)
    assert s["monte_carlo"]["p5"] < s["monte_carlo"]["p95"]


def test_ddm_gordon_formula() -> None:
    out = dividend_discount(
        inputs(), DdmParams(beta=1.0, growth=0.05, terminal_growth=0.03)
    ).payload
    assert out["gordon"] == pytest.approx(out["dps"] * 1.03 / (out["cost_of_equity"] - 0.03))
    h = out["dps"] * 1.03 / (out["cost_of_equity"] - 0.03) + out["dps"] * 2.5 * 0.02 / (
        out["cost_of_equity"] - 0.03
    )
    assert out["h_model"] == pytest.approx(h)


def test_residual_income_and_epv() -> None:
    ri = residual_income(inputs(), WaccParams(beta=1.0)).payload
    assert ri["rows"][-1]["roe"] == pytest.approx(ri["cost_of_equity"])  # ROE fades to k_e
    epv = earnings_power(inputs(), WaccParams(beta=1.0)).payload
    assert epv["franchise_value"] == pytest.approx(epv["market_ev"] - epv["epv"])


def test_point_in_time_ttm_never_uses_unfiled_data() -> None:
    pit = pit_ttm(ALL)
    known = pit["revenue__known"].dropna()
    assert (known >= known.index).all()  # every TTM revenue is known after its period ends
    dates = pd.DatetimeIndex(["2024-10-31", "2024-11-01", "2024-11-04"])
    daily = pit_daily(pit, dates)
    # Apple's FY2024 10-K (which completes Q4 FY2024) was filed on 2024-11-01: TTM revenue
    # changes on that day, not before.
    assert (
        daily["revenue__known"].iloc[0]
        < pd.Timestamp("2024-11-01")
        <= daily["revenue__known"].iloc[2]
    )
    assert daily["revenue"].iloc[0] != pytest.approx(daily["revenue"].iloc[2])


def test_pit_multiples_and_decomposition() -> None:
    out = pit_multiples(inputs(), HistoryParams(years=5)).payload
    assert "pe" in out["stats"]
    assert out["stats"]["pe"]["p10"] <= out["stats"]["pe"]["p50"] <= out["stats"]["pe"]["p90"]
    dec = multiple_decomposition(inputs(), HistoryParams(years=5)).payload["decomposition"]
    assert dec["price_log_return"] == pytest.approx(dec["eps_growth"] + dec["multiple_change"])


def test_football_field_and_iso_value_curve() -> None:
    ff = football_field(inputs(), DcfParams(beta=1.0)).payload
    methods = {b["method"] for b in ff["bars"] if b["low"] is not None}
    assert {"DCF (bear–bull)", "52-week range"} <= methods
    curve = implied_expectations(inputs(), WaccParams(beta=1.0)).payload["curve"]
    point = next(c for c in curve if c["implied_growth"] is not None)
    w = implied_expectations(inputs(), WaccParams(beta=1.0)).payload
    g, roic = point["implied_growth"], point["roic"]
    assert w["nopat"] * (1 - g / roic) / (w["wacc"] - g) == pytest.approx(
        w["enterprise_value"], rel=1e-6
    )


def test_sotp_needs_segments() -> None:
    with pytest.raises(InsufficientDataError):
        sum_of_parts(inputs(), SotpParams())
    out = sum_of_parts(
        inputs(),
        SotpParams.model_validate({"segments": [{"name": "A", "ebitda": 1e10, "multiple": 10}]}),
    ).payload
    assert out["enterprise_value"] == pytest.approx(1e11)
    assert math.isfinite(out["per_share"])


def test_split_adjustment_of_point_in_time_share_counts() -> None:
    from fda_core.analyses.valuation import split_factor

    splits = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2024-06-10"), pd.Timestamp("2021-07-20")], "ratio": [10.0, 4.0]}
    )
    assert split_factor(splits, pd.Timestamp("2020-12-31")) == 40.0
    assert split_factor(splits, pd.Timestamp("2023-12-31")) == 10.0
    assert split_factor(splits, pd.Timestamp("2025-01-31")) == 1.0
    plain = pit_ttm(ALL)
    adjusted = pit_ttm(ALL, pd.DataFrame({"ex_date": [pd.Timestamp("2026-01-15")], "ratio": [2.0]}))
    before = adjusted.index < pd.Timestamp("2026-01-15")
    assert (
        adjusted.loc[before, "shares_outstanding"] == plain.loc[before, "shares_outstanding"] * 2
    ).all()


def test_scorecard_valuation_pillar_and_value_flag() -> None:
    from fda_core.analyses.ratios import FyParams, ScorecardParams, health_scorecard, screen_flags

    out = health_scorecard(inputs(), ScorecardParams()).payload
    assert out["valuation"] == "scored"
    rows = {r["metric"]: r for r in out["drilldown"] if r["pillar"] == "valuation"}
    assert set(rows) == {"pe", "ev_ebitda", "fcf_yield"}
    pe = rows["pe"]
    assert pe["absolute"] == pytest.approx(min(max((pe["value"] - 35) / (12 - 35), 0), 1) * 100)
    assert "valuation" in out["pillars"]
    flags = {f["flag"]: f for f in screen_flags(inputs(), FyParams()).payload["flags"]}
    ev_fcf = current_multiples(inputs(), NoParams()).payload["multiples"]["ev_fcf"]
    assert flags["value"]["value"] == pytest.approx(ev_fcf)
    assert flags["value"]["on"] == (ev_fcf < 15)
    ff = football_field(inputs(), DcfParams(beta=1.0)).payload
    peer = next(b for b in ff["bars"] if b["method"] == "Peer multiples")
    assert "peers" in peer["note"]  # no peer group in the inputs: explained
