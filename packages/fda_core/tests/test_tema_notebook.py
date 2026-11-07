"""Numeric reproduction of Quant Rick's TEMA notebook (`TEMA-TEMPLATE_adjusted6_warmup`).

Reference figures: the owner's review, which re-ran the notebook (yfinance, vectorbt 1.1.1):
`QuantRick-Course/01_Trend_Following_Bootcamp1/Revues_notebooks/TEMA-TEMPLATE_adjusted6_warmup.md`.
Data: TQQQ daily, adjusted, 2018-01-02 → 2026-03-30 (2071 bars), stored once in tests/data.
Protocol: 60/40 split (bar 1242), warm-up 220 bars with the signals before it discarded, fees
0.05 % + slippage 0.05 % per side, the crossing seen at close t and traded at close t+1
(`shift(1)` then vectorbt's same-bar execution = the engine's lag 2).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fda_core import backtest as bt

DATA = Path(__file__).resolve().parents[3] / "tests/data/tqqq_yf_adjusted_2018-01-02_2026-03-30.csv"
WARMUP, SPLIT = 220, 1242

# the series is third-party data kept out of the public copy: these tests run where it exists
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="TQQQ reference series not present")
COSTS = bt.Costs(5, 5)


@pytest.fixture(scope="module")
def close() -> pd.Series:
    frame = pd.read_csv(DATA, index_col=0, parse_dates=True)
    return frame["close"].astype(float)


def in_sample(close: pd.Series, n: tuple[int, int, int], lag: int) -> pd.DataFrame:
    target = bt.tema_events(close, *n, start=WARMUP)
    frame = bt.run_vectorised(close, target, COSTS, WARMUP, lag)
    return frame.iloc[: SPLIT - WARMUP]


def stats(frame: pd.DataFrame) -> tuple[float, float, float]:
    nav = (1 + frame["net"]).cumprod()
    return (
        bt.sharpe(frame["net"].to_numpy()),
        float(nav.iloc[-1] - 1),
        float((nav / nav.cummax() - 1).min()),
    )


def test_sample_and_windows_match_the_notebook(close: pd.Series) -> None:
    assert len(close) == 2071
    assert close.index[SPLIT].date().isoformat() == "2022-12-07"
    frame = in_sample(close, (4, 119, 127), 2)
    assert frame.index[0].date().isoformat() == "2018-11-14"
    assert frame.index[-1].date().isoformat() == "2022-12-06"


def test_best_in_sample_cell_reproduces_the_review(close: pd.Series) -> None:
    """Review: EMA(4,119,127) IS Sharpe 1,104, return 412,6 %, max DD −35,78 %."""
    sr, ret, dd = stats(in_sample(close, (4, 119, 127), 2))
    assert sr == pytest.approx(1.104, abs=0.005)
    assert ret == pytest.approx(4.126, abs=0.01)
    assert dd == pytest.approx(-0.3578, abs=0.002)


def test_execution_delay_table_of_the_review(close: pd.Series) -> None:
    """Review §6.1: close t 0,686 / 132 %; close t+1 (notebook) 1,104 / 413 %; t+2 0,835 / 215 %."""
    table = {lag: stats(in_sample(close, (4, 119, 127), lag)) for lag in (1, 2, 3)}
    assert table[1][0] == pytest.approx(0.686, abs=0.005)
    assert table[1][1] == pytest.approx(1.32, abs=0.02)
    assert table[3][0] == pytest.approx(0.835, abs=0.005)
    assert table[3][1] == pytest.approx(2.15, abs=0.02)


def test_buy_and_hold_in_sample(close: pd.Series) -> None:
    """Review: buy & hold TQQQ IS Sharpe 0,582 (daily adjusted closes may move with later
    dividend adjustments; the tolerance covers that)."""
    r = close.pct_change().iloc[WARMUP:SPLIT].to_numpy()
    assert bt.sharpe(r) == pytest.approx(0.582, abs=0.02)


def test_the_optimum_is_found_again_in_its_neighbourhood(close: pd.Series) -> None:
    """EMA(4,119,127) is the in-sample best of its neighbourhood (the review's top 5 are all EMA1
    = 4 with EMA2/EMA3 around 119-139): the grid's winner, found again on our engine."""
    cells = [
        (n1, n2, n3) for n1 in (4, 7) for n2 in range(113, 126, 3) for n3 in range(121, 140, 3)
    ]
    sharpe = {c: stats(in_sample(close, c, 2))[0] for c in cells}
    assert max(sharpe, key=lambda c: sharpe[c]) == (4, 119, 127)
    # the slower fast EMA does worse on average: the edge rides on the 4-day EMA
    assert np.mean([sharpe[c] for c in cells if c[0] == 7]) < sharpe[(4, 119, 127)]
