"""Golden tests of the portfolio engine (fda_core.portfolio): hand-computed paths and P&L."""

import numpy as np
import pandas as pd
import pytest

from fda_core.portfolio import (
    CASH,
    build_path,
    covariance,
    risk_contributions,
    xirr,
)
from fda_core.registry import InsufficientDataError


def frame(
    close: list[float] | np.ndarray, start: str = "2024-01-02", adj: np.ndarray | None = None
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {"close": close, "adj_close": close if adj is None else adj, "volume": 1e6}, index=index
    )


def ledger(*rows: tuple[object, ...]) -> pd.DataFrame:
    """Rows as (date, type, symbol, quantity, price, currency, fees)."""
    cols = ["trade_date", "type", "symbol", "quantity", "price", "currency", "fees"]
    return pd.DataFrame([dict(zip(cols, (*r, 0.0)[:7], strict=False)) for r in rows])


def weights_def(**extra: object) -> dict[str, object]:
    return {"mode": "weights", "base_currency": "USD", "initial_value": 1000.0} | extra


def test_single_asset_weights_portfolio_equals_the_asset() -> None:
    rng = np.random.default_rng(1)
    a = frame(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    path = build_path(
        weights_def(holdings=[{"symbol": "A", "weight": 1}]), {"A": a}, {"A": "USD"}, {}
    )
    expected = 100 * a["adj_close"] / a["adj_close"].iloc[0]
    assert np.allclose(path.nav.to_numpy(), expected.to_numpy())
    assert path.costs.sum() == 0


def test_monthly_rebalance_resets_weights_and_charges_costs() -> None:
    a = frame(np.linspace(100, 200, 60))  # Jan 2 → Mar 22, 2024
    b = frame(np.full(60, 50.0))
    definition = weights_def(
        holdings=[{"symbol": "A", "weight": 0.5}, {"symbol": "B", "weight": 0.5}],
        rebalance={"type": "calendar", "freq": "M"},
        cost_bps=10,
    )
    path = build_path(definition, {"A": a, "B": b}, {"A": "USD", "B": "USD"}, {})
    month_end = pd.Timestamp("2024-01-31")
    after = path.weights.index[path.weights.index.get_loc(month_end) + 1]
    assert path.weights.loc[after, "A"] == pytest.approx(0.5)  # reset at the January close
    # Cost = 10 bp × traded notional at the rebalance close.
    before = path.weights.loc[month_end]  # start-of-day weights
    v_prev = path.value.shift(1).loc[month_end]
    h_a = v_prev * before["A"] * (a.loc[month_end, "close"] / a["close"].shift(1).loc[month_end])
    h_b = v_prev * before["B"]
    v_pre = h_a + h_b
    traded = abs(h_a - v_pre / 2) + abs(h_b - v_pre / 2)
    assert path.costs.loc[month_end] == pytest.approx(traded * 10 / 1e4)
    assert path.turnover.loc[month_end] == pytest.approx(traded / (2 * v_pre))
    drift = build_path(
        weights_def(holdings=definition["holdings"], rebalance={"type": "none"}),
        {"A": a, "B": b},
        {"A": "USD", "B": "USD"},
        {},
    )
    assert drift.weights["A"].iloc[-1] > 0.6
    with pytest.raises(InsufficientDataError, match="sum"):
        build_path(
            weights_def(holdings=[{"symbol": "A", "weight": 0.7}]), {"A": a}, {"A": "USD"}, {}
        )


def test_cash_and_threshold_policy() -> None:
    a = frame(np.linspace(100, 300, 40))
    definition = weights_def(
        holdings=[
            {"symbol": "A", "weight": 0.6},
            {"symbol": "CASH", "weight": 0.4, "is_cash": True},
        ],
        rebalance={"type": "threshold", "band": 0.05},
    )
    path = build_path(definition, {"A": a}, {"A": "USD"}, {})
    assert CASH in path.weights
    assert (path.weights["A"] <= 0.65 + 0.02).all()
    assert path.turnover.gt(0).sum() >= 2


def test_ledger_replay_twr_fifo_and_flows() -> None:
    a = frame([50.0, 55, 60, 60, 66])
    days = a.index
    tx = ledger(
        (days[0], "DEPOSIT", None, 1000, 1, "USD"),
        (days[0], "BUY", "A", 10, 50, "USD"),
        (days[1], "BUY", "A", 10, 55, "USD"),
        (days[2], "DEPOSIT", None, 500, 1, "USD"),
        (days[3], "SELL", "A", 15, 60, "USD", 3),
        (days[4], "DIVIDEND", "A", 4, 1, "USD"),
    )
    path = build_path(
        {"mode": "transactions", "base_currency": "USD", "transactions": tx, "lot_method": "fifo"},
        {"A": a},
        {"A": "USD"},
        {},
    )
    v = [1000.0, 1050.0, 1650.0, 1647.0, 1681.0]  # cash + 5 shares × close (+ dividend)
    assert np.allclose(path.value.to_numpy(), v)
    r = [1050 / 1000, (1650 - 500) / 1050, 1647 / 1650, 1681 / 1647]
    assert path.nav.iloc[-1] == pytest.approx(100 * np.prod(r))
    # FIFO: 10 @ 50 then 5 @ 55 closed at (15×60 − 3)/15.
    unit = (15 * 60 - 3) / 15
    pnl = sum(x["pnl"] for x in path.realised)
    assert pnl == pytest.approx(10 * (unit - 50) + 5 * (unit - 55))
    assert path.lots == [
        {"symbol": "A", "opened": days[1].date().isoformat(), "quantity": 5, "unit_cost": 55}
    ]
    assert path.income[0]["amount"] == 4
    avg = build_path(
        {"mode": "transactions", "transactions": tx, "lot_method": "average"},
        {"A": a},
        {"A": "USD"},
        {},
    )
    assert sum(x["pnl"] for x in avg.realised) == pytest.approx(15 * (unit - 52.5))
    # Reconciliation: Δvalue − net flows = realised + Δunrealised + income − fees on the cash side.
    unrealised = 5 * 66 - 5 * 55
    assert path.value.iloc[-1] - 1500 == pytest.approx(pnl + unrealised + 4)


def test_selling_more_than_held_is_refused() -> None:
    a = frame([10.0, 11])
    tx = ledger((a.index[0], "SELL", "A", 1, 10, "USD"))
    with pytest.raises(InsufficientDataError, match="exceeds"):
        build_path({"mode": "transactions", "transactions": tx}, {"A": a}, {"A": "USD"}, {})


def test_split_adjusts_old_trades_and_fx_converts() -> None:
    # Stored prices are split-adjusted: a 4:1 split after the trade turns 1 @ 400 into 4 @ 100.
    a = frame([100.0, 101, 102])
    splits = pd.DataFrame({"ex_date": [a.index[1].date()], "action": ["split"], "ratio": [4.0]})
    tx = ledger(
        (a.index[0], "DEPOSIT", None, 400, 1, "EUR"),
        (a.index[0], "BUY", "A", 1, 400, "EUR"),
    )
    eurusd = pd.Series([1.10, 1.10, 1.20], index=a.index)
    path = build_path(
        {"mode": "transactions", "base_currency": "USD", "transactions": tx},
        {"A": a},
        {"A": "EUR"},
        {"EUR": eurusd},
        {"A": splits},
    )
    assert path.quantities["A"].iloc[-1] == 4
    assert path.value.iloc[-1] == pytest.approx(4 * 102 * 1.2)
    assert path.returns["A"].iloc[-1] == pytest.approx(102 * 1.2 / (101 * 1.1) - 1)
    assert path.local_returns["A"].iloc[-1] == pytest.approx(102 / 101 - 1)
    with pytest.raises(InsufficientDataError, match="EURUSD=X"):
        build_path(
            {"mode": "transactions", "base_currency": "USD", "transactions": tx},
            {"A": a},
            {"A": "EUR"},
            {},
        )


def test_xirr_golden() -> None:
    t0 = pd.Timestamp("2024-01-01")
    assert xirr([t0, t0 + pd.Timedelta(days=365)], [-1000, 1100]) == pytest.approx(0.10)
    flows = [-1000, -500, 1800]
    dates = [t0, t0 + pd.Timedelta(days=182), t0 + pd.Timedelta(days=365)]
    rate = xirr(dates, flows)
    assert rate is not None
    years = np.array([(d - t0).days / 365 for d in dates])
    assert np.sum(np.array(flows) / (1 + rate) ** years) == pytest.approx(0, abs=1e-6)
    assert xirr([t0], [100]) is None


def test_risk_contributions_sum_to_sigma_for_every_estimator() -> None:
    rng = np.random.default_rng(5)
    x = pd.DataFrame(
        rng.multivariate_normal([0, 0, 0], [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 2]], 400) / 100
    )
    w = np.array([0.5, 0.3, 0.2])
    for est in ("sample", "ledoit_wolf", "oas", "ewma"):
        cov = covariance(x, est)  # type: ignore[arg-type]
        assert np.allclose(cov, cov.T)
        sigma, _, contrib = risk_contributions(w, cov)
        assert contrib.sum() == pytest.approx(sigma)
    assert np.allclose(covariance(x, "sample"), np.cov(x.to_numpy(), rowvar=False))


def test_ledoit_wolf_matches_sklearn() -> None:
    sk = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(120, 6)))
    n = len(x)
    ours_lw = covariance(x, "ledoit_wolf") * (n - 1) / n
    ours_oas = covariance(x, "oas") * (n - 1) / n
    assert np.allclose(ours_lw, sk.LedoitWolf().fit(x).covariance_)
    assert np.allclose(ours_oas, sk.OAS().fit(x).covariance_, atol=1e-3)
