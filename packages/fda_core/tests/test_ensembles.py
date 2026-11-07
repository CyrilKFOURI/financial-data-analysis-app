"""MACD-V and Donchian-ensemble templates, signal ensembles and the shadow test (TR-025)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fda_core import backtest as bt
from fda_core import robustness as rb
from fda_core.analyses.ensembles import EnsembleParams, signal_ensembles
from fda_core.registry import AnalysisInputs, InsufficientDataError

DAYS = pd.bdate_range("2010-01-04", periods=2000)


def walk(seed: int = 4, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.011, len(DAYS)))), index=DAYS)


def test_macd_v_on_a_linear_ramp() -> None:
    """Close = 100 + t: |Δclose| = 1 so the ATR is 1, and an EMA of span n lags a ramp by
    (n − 1)/2, so MACD-V converges to ((26 − 1) − (12 − 1))/2 × 100 = 700."""
    close = pd.Series(100.0 + np.arange(600), index=DAYS[:600])
    value = bt.macd_v(close, 12, 26)
    assert value.iloc[-1] == pytest.approx(700.0, rel=1e-6)
    assert bt.signal("macd_v", close, {"fast": 12, "slow": 26, "threshold": 50}).iloc[-1] == 1.0


def test_donchian_ensemble_is_a_share_of_channels() -> None:
    close = walk()
    target = bt.signal("donchian_ensemble", close, {"shortest": 10, "levels": 4})
    assert target.between(0, 1).all()
    assert set(np.round(target.unique() * 4)) <= {0, 1, 2, 3, 4}
    # a steady rise puts every channel long
    ramp = pd.Series(100.0 + np.arange(400), index=DAYS[:400])
    assert bt.signal("donchian_ensemble", ramp, {"shortest": 10, "levels": 3}).iloc[-1] == 1.0


def test_template_parameters_are_bounded() -> None:
    """Grids come from users: an absurd level count or window is refused, not looped over."""
    close = walk()
    with pytest.raises(InsufficientDataError):
        bt.signal("donchian_ensemble", close, {"shortest": 10, "levels": 60})
    with pytest.raises(InsufficientDataError):
        bt.signal("donchian_ensemble", close, {"shortest": 1, "levels": 3})
    with pytest.raises(InsufficientDataError):
        bt.signal("donchian_ensemble", close, {"shortest": 400, "levels": 4})
    with pytest.raises(InsufficientDataError):
        bt.signal("macd_v", close, {"fast": 26, "slow": 12, "threshold": 0})


def test_combine_rules() -> None:
    signals = pd.DataFrame({"a": [1, 1, 0, 0], "b": [1, 0, 0, 1], "c": [1, 0, 0, 0.5]})
    assert rb.combine(signals, "or").tolist() == [1, 1, 0, 1]
    assert rb.combine(signals, "and").tolist() == [1, 0, 0, 0]
    assert rb.combine(signals, "vote").tolist() == [1, 0, 0, 1]
    assert rb.combine(signals, "mean").tolist() == pytest.approx([1, 1 / 3, 0, 0.5])
    with pytest.raises(ValueError, match="Unknown ensemble rule"):
        rb.combine(signals, "xor")


def frame_for(position: np.ndarray, asset: np.ndarray, costs: bt.Costs) -> pd.DataFrame:
    traded = np.abs(np.diff(position, prepend=0.0))
    return pd.DataFrame(
        {"position": position, "asset": asset, "net": position * asset - traded * costs.per_unit},
        index=DAYS[: len(asset)],
    )


def test_shadow_test_separates_timing_from_noise() -> None:
    rng = np.random.default_rng(9)
    asset = rng.normal(0.0002, 0.01, 1500)
    costs = bt.Costs(1, 1)
    # a position that knows the sign of the day's return has timing; a random one has none
    timed = frame_for((asset > 0).astype(float), asset, costs)
    noise = frame_for((rng.random(1500) > 0.5).astype(float), asset, costs)
    good = rb.shadow_test(timed, costs, 252.0, n=100)
    bad = rb.shadow_test(noise, costs, 252.0, n=100)
    assert good["p_value"] == pytest.approx(1 / 101)
    assert bad["p_value"] > 0.05
    # deterministic with its seed
    assert rb.shadow_test(noise, costs, 252.0, n=100) == bad


def test_analysis_members_ensembles_and_in_sample_choice() -> None:
    close = walk()
    strat: dict[str, Any] = {
        "id": 1,
        "name": "t",
        "hypothesis": "Trend persistence in a drifting series, tested before reading results.",
        "trials": 0,
        "spec": {
            "symbol": "WALK",
            "template": "ema_cross",
            "grid": {"fast": [10, 20, 30], "slow": [100, 150]},
            "params": {"fast": 20, "slow": 150},
            "short": False,
            "overlays": [],
            "costs": {"fee_bps": 5, "slippage_bps": 5},
            "warmup": 220,
            "is_fraction": 0.6,
        },
    }
    out = signal_ensembles(
        AnalysisInputs(data={"strategy": strat, "prices": pd.DataFrame({"adj_close": close})}),
        EnsembleParams(members=3, shadows=60),
    ).payload
    names = [r["series"] for r in out["rows"]]
    assert names[3:] == ["ensemble or", "ensemble and", "ensemble vote", "ensemble mean"]
    assert len(names) == 7
    # members are ranked by in-sample Sharpe
    is_sr = [r["is_sharpe"] for r in out["rows"][:3]]
    assert is_sr == sorted(is_sr, reverse=True)
    assert all(0 < r["p_value"] <= 1 for r in out["rows"])
