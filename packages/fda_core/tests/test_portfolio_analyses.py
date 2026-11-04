"""Golden tests for P-001…P-026 and C-001…C-007 on synthetic portfolios."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses import portfolio as pa
from fda_core.analyses.price import NoParams as PriceNoParams
from fda_core.analyses.price import volatility
from fda_core.portfolio import PortfolioInputs, build_path
from fda_core.registry import REGISTRY, AnalysisInputs, InsufficientDataError, NotApplicableError

N = pa.NoParams()


def prices(
    seed: int, n: int = 800, start: str = "2021-01-04", drift: float = 0.0003, vol: float = 0.012
):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"close": close, "adj_close": close, "volume": 1e6}, index=index)


META = {
    "AAA": {
        "asset_class": "equity",
        "asset_subclass": "common",
        "sector": "Technology",
        "country": "United States",
        "currency": "USD",
        "info": {"trailingPE": 20.0, "operatingMargins": 0.30, "marketCap": 3e11},
    },
    "BBB": {
        "asset_class": "equity",
        "asset_subclass": "common",
        "sector": "Energy",
        "country": "France",
        "currency": "EUR",
        "info": {"trailingPE": 10.0, "operatingMargins": 0.10, "marketCap": 5e9},
    },
    "CCC": {
        "asset_class": "equity",
        "asset_subclass": "etf",
        "sector": None,
        "country": None,
        "currency": "USD",
        "info": {"trailingPE": 25.0},
    },
}


def make(
    weights: dict[str, float],
    frames: dict[str, pd.DataFrame] | None = None,
    name: str = "P",
    fx: dict[str, pd.Series] | None = None,
    **definition: object,
) -> PortfolioInputs:
    frames = frames or {s: prices(i) for i, s in enumerate(weights) if s != "CASH"}
    holdings = [{"symbol": s, "weight": w, "is_cash": s == "CASH"} for s, w in weights.items()]
    d = {"mode": "weights", "name": name, "base_currency": "USD", "holdings": holdings} | definition
    currencies = {s: META.get(s, {}).get("currency", "USD") for s in frames}
    fx = fx if fx is not None else {"EUR": pd.Series(1.1, index=next(iter(frames.values())).index)}
    path = build_path(d, frames, currencies, fx)
    return PortfolioInputs(
        path=path,
        definition=d,
        meta={s: META.get(s, {}) for s in frames},
        prices=frames,
        fx_last={"EUR": 1.1, "USD": 1.0},
    )


def inp(pf: PortfolioInputs, **data: object) -> AnalysisInputs:
    return AnalysisInputs(
        data={"portfolio": pf} | data, conventions={"range": "MAX", "annualisation": 252}
    )


def test_every_portfolio_id_is_registered() -> None:
    ids = [
        f"P-{i:03d}"
        for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
    ]
    ids += [f"C-{i:03d}" for i in (1, 2, 3, 4, 5, 7)]
    assert all(i in REGISTRY for i in ids)
    assert "portfolio" in REGISTRY.get("R-001").applies_to  # Subject abstraction


def test_single_asset_portfolio_matches_the_asset() -> None:
    frame = prices(1)
    pf = make({"AAA": 1.0}, {"AAA": frame})
    nav = pd.DataFrame(
        {c: pf.path.nav for c in ("open", "high", "low", "close", "adj_close")} | {"volume": np.nan}
    )
    conv = {"range": "MAX", "annualisation": 252, "price_field": "adj_close"}
    v_pf = volatility(AnalysisInputs({"prices": nav}, conv), PriceNoParams()).payload
    v_a = volatility(AnalysisInputs({"prices": frame}, conv), PriceNoParams()).payload
    assert v_pf["value"] == pytest.approx(v_a["value"])
    d = pa.risk_decomposition(inp(pf), pa.RiskParams()).payload
    assert d["volatility"] == pytest.approx(frame["adj_close"].pct_change().std() * math.sqrt(252))
    assert pa.twr(inp(pf), N).payload["twr"] == pytest.approx(
        frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
    )
    assert pa.mwr(inp(pf), N).payload["gap"] == pytest.approx(0, abs=1e-6)


def test_risk_decomposition_and_component_es_sum_exactly() -> None:
    pf = make({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    for est in ("sample", "ledoit_wolf", "oas", "ewma"):
        d = pa.risk_decomposition(inp(pf), pa.RiskParams(estimator=est)).payload
        assert sum(r["risk_contribution"] for r in d["rows"]) == pytest.approx(d["volatility"])
        assert sum(r["risk_share"] for r in d["rows"]) == pytest.approx(1)
    es = pa.component_es(inp(pf), pa.EsParams()).payload
    assert sum(r["component_es"] for r in es["rows"]) == pytest.approx(es["es"])
    assert es["es"] >= es["var"]
    dr = pa.diversification_ratio(inp(pf), pa.RiskParams()).payload
    assert dr["diversification_ratio"] > 1
    assert {r["estimator"] for r in pa.covariance_estimators(inp(pf), N).payload["rows"]} == {
        "sample",
        "ledoit_wolf",
        "oas",
        "ewma",
    }
    assert 0 < pa.absorption_ratio(inp(pf), pa.AbsorptionParams()).payload["current"] <= 1


def test_concentration_contribution_allocations() -> None:
    pf = make({"AAA": 0.5, "BBB": 0.3, "CASH": 0.2}, rebalance={"type": "calendar", "freq": "M"})
    c = pa.concentration(inp(pf), N).payload
    w = pf.end_weights()
    assert c["hhi"] == pytest.approx(float((w**2).sum()))
    assert c["effective_n"] == pytest.approx(1 / c["hhi"])
    contrib = pa.contribution(inp(pf), N).payload
    simple = float((pf.path.nav.pct_change().dropna()).sum())
    assert contrib["sum"] == pytest.approx(simple)
    alloc = pa.allocations(inp(pf), N).payload["breakdowns"]
    assert sum(g["weight"] for g in alloc["sector"]) == pytest.approx(1)
    assert {g["group"] for g in alloc["region"]} == {"North America", "Europe", "Cash"}
    assert {g["group"] for g in alloc["market_cap"]} == {"mega", "mid", "Cash"}
    drift = pa.weights_mode(inp(pf), N).payload
    assert drift["rebalances"] > 30
    with pytest.raises(NotApplicableError):
        pa.ledger(inp(pf), N)


def test_brinson_linking_and_active_share() -> None:
    frames = {"AAA": prices(1), "BBB": prices(2), "CCC": prices(3)}
    pf = make({"AAA": 0.6, "BBB": 0.4}, frames)
    bm = make({"AAA": 0.3, "BBB": 0.3, "CCC": 0.4}, frames, name="Bench")
    out = pa.brinson(inp(pf, compare_portfolio=bm), pa.CompareParams()).payload
    total = sum(r["total"] for r in out["rows"])
    assert total == pytest.approx(out["portfolio_return"] - out["benchmark_return"], abs=1e-10)
    same = pa.brinson(inp(pf, compare_portfolio=pf), pa.CompareParams()).payload
    assert all(abs(r["total"]) < 1e-12 for r in same["rows"])
    a = pa.active_share(inp(pf, compare_portfolio=bm), pa.CompareParams()).payload
    wp, wb = pf.end_weights(), bm.end_weights()
    names = set(wp.index) | set(wb.index)
    assert a["active_share"] == pytest.approx(
        0.5 * sum(abs(wp.get(s, 0) - wb.get(s, 0)) for s in names)
    )
    assert pa.active_share(inp(pf, compare_portfolio=pf), pa.CompareParams()).payload[
        "tracking_error_ex_ante"
    ] == pytest.approx(0)
    ov = pa.overlap(inp(pf, compare_portfolio=bm), pa.CompareParams()).payload
    assert ov["overlap"] == pytest.approx(
        sum(min(wp.get(s, 0), wb.get(s, 0)) for s in ("AAA", "BBB"))
    )
    with pytest.raises(InsufficientDataError, match="benchmark portfolio"):
        pa.brinson(inp(pf), pa.CompareParams())


def test_brinson_fachler_single_period_golden() -> None:
    # Classic textbook case: two sectors.
    eff = pa.brinson_fachler(
        {"A": 0.7, "B": 0.3}, {"A": 0.10, "B": 0.02}, {"A": 0.5, "B": 0.5}, {"A": 0.08, "B": 0.04}
    )
    b = 0.5 * 0.08 + 0.5 * 0.04
    assert eff["A"]["allocation"] == pytest.approx(0.2 * (0.08 - b))
    assert eff["A"]["selection"] == pytest.approx(0.5 * 0.02)
    assert eff["A"]["interaction"] == pytest.approx(0.2 * 0.02)
    total = sum(sum(v.values()) for v in eff.values())
    assert total == pytest.approx((0.7 * 0.10 + 0.3 * 0.02) - b)


def test_look_through_currency_liquidity() -> None:
    pf = make({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, rebalance={"type": "none"})
    lt = {r["metric"]: r for r in pa.look_through(inp(pf), N).payload["rows"]}
    w = pf.end_weights()
    covered = w["AAA"] + w["BBB"]
    assert lt["pe"]["value"] == pytest.approx(covered / (w["AAA"] / 20 + w["BBB"] / 10))
    assert lt["operating_margin"]["value"] == pytest.approx(
        (w["AAA"] * 0.3 + w["BBB"] * 0.1) / covered
    )
    assert lt["pe"]["covered_weight"] == pytest.approx(covered)  # the ETF is excluded
    # A EUR holding with a rising EUR: FX contribution positive, local part unaffected.
    frames = {"AAA": prices(1), "BBB": prices(2)}
    eurusd = pd.Series(np.linspace(1.0, 1.2, 800), index=frames["BBB"].index)
    fx_pf = make({"AAA": 0.5, "BBB": 0.5}, frames, fx={"EUR": eurusd})
    cur = {r["currency"]: r for r in pa.currency_exposure(inp(fx_pf), N).payload["rows"]}
    assert cur["EUR"]["fx_contribution"] > 0
    assert cur["USD"]["fx_contribution"] == pytest.approx(0)
    liq = pa.liquidity(inp(pf), pa.LiquidityParams()).payload
    row = next(r for r in liq["rows"] if r["symbol"] == "AAA")
    assert row["days"] == pytest.approx(row["value"] / (0.2 * row["adv_value"]))


def test_scenarios_factor_betas_and_income() -> None:
    frames = {
        "AAA": prices(1, n=5000, start="2006-01-02"),
        "BBB": prices(2, n=5000, start="2006-01-02"),
    }
    pf = make({"AAA": 0.5, "BBB": 0.5}, frames)
    sc = pa.scenario_replay(inp(pf), N).payload["scenarios"]
    gfc = next(s for s in sc if s["scenario"] == "GFC")
    w = pf.end_weights()
    exp = sum(
        w[s]
        * (
            f["adj_close"].asof(pd.Timestamp(gfc["end"]))
            / f["adj_close"].asof(pd.Timestamp(gfc["start"]))
            - 1
        )
        for s, f in frames.items()
    )
    assert gfc["portfolio_return"] == pytest.approx(exp)
    # Portfolio built as 1.5 × an equity factor: beta ≈ 1.5.
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=900)
    mkt = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 900)), index=idx)
    asset = 100 * np.cumprod(1 + 1.5 * mkt.pct_change().fillna(0).to_numpy())
    f = pd.DataFrame({"close": asset, "adj_close": asset, "volume": 1e6}, index=idx)
    one = make({"AAA": 1.0}, {"AAA": f})
    out = pa.factor_shocks(inp(one, factors={"equity": mkt}), pa.FactorParams()).payload
    assert out["rows"][0]["beta"] == pytest.approx(1.5, abs=0.02)
    assert out["combined_impact"] == pytest.approx(1.5 * -0.20, abs=0.01)
    assert "rates_10y" in out["missing_factors"]
    with pytest.raises(InsufficientDataError, match="factor"):
        pa.factor_shocks(inp(one), pa.FactorParams())
    # Weights mode income: implied quantity × stored dividends.
    a = prices(1)
    divs = pd.DataFrame(
        {
            "ex_date": [a.index[-200].date(), a.index[-50].date()],
            "action": ["dividend"] * 2,
            "amount": [1.0, 1.0],
        }
    )
    pf_inc = make({"AAA": 1.0}, {"AAA": a})
    pf_inc.actions = {"AAA": divs}
    income = pa.income(inp(pf_inc), N).payload
    qty = pf_inc.path.holdings_value["AAA"].iloc[-1] / a["close"].iloc[-1]
    assert income["forward_12m"] == pytest.approx(qty * 2.0)
    assert income["trailing_12m"] > 0


def test_interactions() -> None:
    pf = make({"AAA": 0.4, "BBB": 0.3, "CCC": 0.3})
    cm = pa.correlation_matrix(inp(pf), pa.CorrParams()).payload
    m = np.array(cm["matrix"])
    assert np.allclose(np.diag(m), 1)
    assert np.allclose(m, m.T)
    assert sorted(cm["labels"]) == ["AAA", "BBB", "CCC"]
    assert (
        pa.correlation_matrix(inp(pf), pa.CorrParams(method="kendall", freq="W")).payload[
            "observations"
        ]
        < 200
    )
    roll = pa.rolling_pair(inp(pf), pa.PairParams()).payload
    assert (roll["a"], roll["b"]) == ("AAA", "BBB")
    st = pa.stress_correlation(inp(pf), N).payload
    assert st["driver"] == "portfolio"
    assert st["stress_count"] == pytest.approx(0.1 * 799, abs=2)
    rr = pa.risk_return_map(inp(pf), N).payload
    assert len(rr["points"]) == 3
    stab = pa.correlation_stability(inp(pf), N).payload
    assert len(stab["changed"]) == 3
    td = pa.tail_dependence(inp(pf), pa.TailParams()).payload
    assert np.allclose(np.array(td["matrix"]), np.array(td["matrix"]).T)
    assert all(0 <= p["lambda"] <= 1 for p in td["top"])
