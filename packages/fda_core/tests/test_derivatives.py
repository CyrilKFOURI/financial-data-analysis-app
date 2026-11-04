"""Golden tests for the DV analyses on a chain priced from a known SVI smile."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.derivatives import (
    ChainParams,
    PositioningParams,
    SmileParams,
    atm_iv_at,
    enrich_chain,
    fit_expiries,
    iv_surface,
    max_pain,
    option_chain,
    positioning,
    smile_skew,
    straddle_moves,
)
from fda_core.options import Svi, bsm_price
from fda_core.registry import AnalysisInputs

AS_OF = date(2026, 10, 23)
SPOT, R, Q = 100.0, 0.04, 0.01


def smile(t: float) -> Svi:
    return Svi(a=0.02 * t, b=0.12 * math.sqrt(t), rho=-0.5, m=0.0, s=0.15)


def chain() -> pd.DataFrame:
    rows = []
    for days in (21, 49, 91, 182):
        t = days / 365
        forward = SPOT * math.exp((R - Q) * t)
        for strike in np.arange(70.0, 131.0, 2.5):
            iv = float(smile(t).iv(np.array([math.log(strike / forward)]), t)[0])
            for kind in ("call", "put"):
                price = float(bsm_price(kind, SPOT, strike, t, R, Q, iv))
                rows.append(
                    {
                        "expiry": pd.Timestamp(AS_OF) + pd.Timedelta(days=days),
                        "kind": kind,
                        "strike": strike,
                        "bid": price * 0.999,
                        "ask": price * 1.001,
                        "last": price,
                        "volume": 100.0 if strike != 100 else 5000.0,
                        "open_interest": 1000.0 + strike,
                        "provider_iv": iv,
                        "contract": f"X{days}{kind}{strike}",
                        "last_trade": pd.Timestamp("2026-10-23", tz="UTC"),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["expiry"] = frame["expiry"].dt.date
    return frame


def inputs() -> AnalysisInputs:
    return AnalysisInputs(
        {"options": {"as_of": AS_OF, "spot": SPOT, "chain": chain()}},
        {"rf": R, "dividend_yield": Q, "price_field": "adj_close"},
    )


def test_enrichment_recovers_the_smile_and_fits_svi() -> None:
    frame = enrich_chain(chain(), SPOT, AS_OF, R, Q)
    sample = frame.loc[(frame["strike"] == 90) & (frame["kind"] == "put")].iloc[0]
    expected = float(smile(sample["t"]).iv(np.array([sample["k"]]), sample["t"])[0])
    assert sample["iv"] == pytest.approx(expected, abs=1e-9)  # mid = model price
    fits = fit_expiries(frame)
    assert len(fits) == 4
    assert all(rmse < 1e-4 for _, rmse, _ in fits.values())
    atm_49 = math.sqrt(smile(49 / 365).w(np.array([0.0]))[0] / (49 / 365))
    assert atm_iv_at(fits, 49) == pytest.approx(atm_49, abs=1e-3)


def test_chain_smile_and_surface_payloads() -> None:
    chain_out = option_chain(inputs(), ChainParams(strikes=4)).payload
    assert chain_out["expiry"] == "2026-11-13"  # first expiry ≥ 7 days
    assert len(chain_out["rows"]) == 9
    atm = next(r for r in chain_out["rows"] if r["strike"] == 100)
    # put-call delta parity; 1e-6 absorbs the platform-dependent last digits of the norm CDF
    assert atm["c_delta"] - atm["p_delta"] == pytest.approx(math.exp(-Q * 21 / 365), abs=1e-6)
    skew = smile_skew(inputs(), SmileParams()).payload
    assert all(r["rr_25d"] < 0 for r in skew["rows"])  # negative rho: puts richer
    surface = iv_surface(inputs(), SmileParams()).payload
    assert surface["arbitrage_free"] is True
    assert len(surface["grid"]["values"]) == 4


def test_positioning_max_pain_and_straddle() -> None:
    frame = pd.DataFrame(
        {
            "kind": ["call", "call", "put", "put"],
            "strike": [90.0, 110.0, 90.0, 110.0],
            "open_interest": [100.0, 100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert max_pain(frame) in (90.0, 110.0)
    out = positioning(inputs(), PositioningParams()).payload
    assert out["put_call_oi"] == pytest.approx(1.0)
    assert any(u["strike"] == 100 for u in out["unusual"])  # 5000 traded vs 1100 OI
    moves = straddle_moves(enrich_chain(chain(), SPOT, AS_OF, R, Q), SPOT)
    first = moves[0]
    assert first["atm_strike"] == 100
    sigma = math.sqrt(
        smile(21 / 365).w(np.array([math.log(100 / (SPOT * math.exp((R - Q) * 21 / 365)))]))[0]
        / (21 / 365)
    )
    assert first["move_pct"] == pytest.approx(0.8 * sigma * math.sqrt(21 / 365), rel=0.05)
