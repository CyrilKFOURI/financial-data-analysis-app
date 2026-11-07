"""Multi-asset Carver book with IDM (TR-026): invariants, no look-ahead, the analysis."""

import numpy as np
import pandas as pd
import pytest

from fda_core import backtest as bt
from fda_core import carver
from fda_core.analyses.carver_portfolio import BookParams, carver_book
from fda_core.portfolio import PortfolioInputs, build_path
from fda_core.registry import AnalysisInputs, InsufficientDataError

DAYS = pd.bdate_range("2012-01-02", periods=1500)


def returns(seed: int, n: int = 3, corr: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, len(DAYS))
    cols = {}
    for i in range(n):
        own = rng.normal(0.0003, 0.01, len(DAYS))
        cols[f"A{i}"] = np.sqrt(corr) * common + np.sqrt(1 - corr) * own
    return pd.DataFrame(cols, index=DAYS)


def test_idm_of_uncorrelated_equal_weights_is_root_n() -> None:
    sub = returns(1, n=4)
    idm = carver.idm_path(sub, np.full(4, 0.25))
    assert idm.iloc[: carver.FDM_MIN_OBS - 1].eq(1.0).all()
    assert idm.iloc[-1] == pytest.approx(2.0, abs=0.15)  # √4, up to sampling noise
    # perfectly correlated sub-systems do not diversify
    same = pd.DataFrame({c: sub["A0"] for c in ("x", "y")})
    assert carver.idm_path(same, np.array([0.5, 0.5])).iloc[-1] == pytest.approx(1.0)


def test_book_has_no_look_ahead_and_respects_the_leverage_cap() -> None:
    r = returns(2)
    w = np.full(3, 1 / 3)

    def size(price: pd.Series) -> pd.Series:
        return carver.position(price, 0.2, 0.1)

    pos, _ = carver.book_positions(r, w, size, 1.5)
    assert pos.abs().sum(axis=1).max() <= 1.5 + 1e-12
    cut = 1000
    altered = r.copy()
    altered.iloc[cut:] *= -3.0
    pos2, _ = carver.book_positions(altered, w, size, 1.5)
    pd.testing.assert_frame_equal(pos.iloc[:cut], pos2.iloc[:cut])


def test_one_asset_book_is_the_single_asset_engine() -> None:
    r = returns(3, n=1)
    price = (1 + r["A0"]).cumprod()
    target = carver.position(price, 0.2, 0.0)
    single = bt.run_vectorised(price, target, bt.Costs(5, 0), 261)
    book = carver.run_book(r, target.to_frame("A0"), 5 / 10_000, 261)
    assert book["net"].to_numpy() == pytest.approx(single["net"].to_numpy(), abs=1e-12)


def portfolio(weights: dict[str, float], frames: dict[str, pd.DataFrame]) -> PortfolioInputs:
    holdings = [{"symbol": s, "weight": w, "is_cash": s == "CASH"} for s, w in weights.items()]
    d = {"mode": "weights", "name": "B", "base_currency": "USD", "holdings": holdings}
    path = build_path(d, frames, dict.fromkeys(frames, "USD"), {})
    return PortfolioInputs(path=path, definition=d, meta={}, prices=frames)


def frames_of(r: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for c in r.columns:
        close = 100 * (1 + r[c]).cumprod()
        out[c] = pd.DataFrame({"close": close, "adj_close": close, "volume": 1e6})
    return out


def test_analysis_compares_the_book_with_hold_and_no_trend() -> None:
    r = returns(4, corr=0.2)
    pf = portfolio({"A0": 0.4, "A1": 0.3, "A2": 0.2, "CASH": 0.1}, frames_of(r))
    out = carver_book(AnalysisInputs(data={"portfolio": pf}), BookParams()).payload
    assert [row["series"] for row in out["rows"]] == [
        "carver_book",
        "vol_targeted_hold",
        "buy_and_hold",
    ]
    # cash is not an instrument; weights are renormalised over the instruments
    assert [x["symbol"] for x in out["weights"]] == ["A0", "A1", "A2"]
    assert sum(x["weight"] for x in out["weights"]) == pytest.approx(1.0)
    assert 1.0 <= out["idm_last"] <= carver.IDM_MAX
    assert out["lines"]["carver_book"]
    assert "Model output" in out["label"]


def test_analysis_needs_two_instruments() -> None:
    r = returns(5, n=1)
    pf = portfolio({"A0": 1.0}, frames_of(r))
    with pytest.raises(InsufficientDataError, match="at least 2 holdings"):
        carver_book(AnalysisInputs(data={"portfolio": pf}), BookParams())
