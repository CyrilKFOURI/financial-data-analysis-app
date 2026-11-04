"""Carver forecast engine diagnostics & prop-firm challenge (ANALYTICS_CATALOG TR-018, TR-019;
module M26). Source: Quantrick bootcamp 2 and the owner's out-of-sample review of it."""

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core import backtest as bt
from fda_core import carver
from fda_core.analyses.backtesting import LABEL, _close, _evaluate, _strategy, _target, _validation
from fda_core.analyses.price import PRICE_SUBJECTS, _prices
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

# Calendar stress windows of the review; shown only when the data covers them after warm-up
CRISES = {
    "2000-2002": ("2000-01-01", "2002-12-31"),
    "2008": ("2008-01-01", "2008-12-31"),
    "2020 Q1": ("2020-01-01", "2020-03-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sharpe_se(sr: float, years: float) -> float | None:
    """Standard error of an annualised Sharpe ratio (Lo 2002, i.i.d.): √((1 + SR²/2) / T)."""
    return math.sqrt((1 + sr * sr / 2) / years) if years > 0 and math.isfinite(sr) else None


def _row(name: str, frame: pd.DataFrame, cut: pd.Timestamp, periods: float) -> dict[str, Any]:
    is_part, oos = frame.loc[frame.index < cut], frame.loc[frame.index >= cut]
    perf = bt.performance(oos, periods=round(periods))
    oos_sr = bt.sharpe(oos["net"].to_numpy(), round(periods))
    row: dict[str, Any] = {
        "variant": name,
        "is_sharpe": _f(bt.sharpe(is_part["net"].to_numpy(), round(periods))),
        "oos_sharpe": _f(oos_sr),
        "oos_sharpe_se": _f(_sharpe_se(oos_sr, len(oos) / periods)),
        "oos_cagr": _f(perf["cagr"]),
        "oos_volatility": _f(perf["volatility"]),
        "oos_max_drawdown": _f(perf["max_drawdown"]),
        "turnover_per_year": _f(perf["turnover_per_year"]),
    }
    for label, (a, b) in CRISES.items():
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        covered = frame.index[0] <= start and frame.index[-1] >= end
        window = frame.loc[(frame.index >= start) & (frame.index <= end), "net"]
        row[f"crisis_{label}"] = _f(float((1 + window).prod() - 1)) if covered else None
    return row


class CarverParams(BaseModel):
    is_fraction: float = Field(0.6, ge=0.3, le=0.9)
    vol_target: float = Field(0.20, gt=0, le=1.0)
    buffer: float = Field(0.10, ge=0, le=0.5)
    fee_bps: float = Field(5.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0)


@analysis(
    id="TR-018",
    version="1.0.0",
    title="Carver forecast engine",
    description=(
        "The four Carver rules (EWMAC 4 speeds, breakout 4 windows, acceleration, skew) with "
        "their published scalars: average |forecast| (target 10), share of bars at the ±20 cap, "
        "correlations, point-in-time FDM. Then the engine (weights summing to 1, F/10 · τ/σ, "
        "long-only, buffer) against two benchmarks: buy & hold and the same volatility sizing "
        "without trend — the comparison that tells whether the trend rules add anything. Sharpe "
        "with its standard error, stress windows, DSR and PBO over the template's default grid."
    ),
    applies_to=PRICE_SUBJECTS,
    params=CarverParams,
    result_kind="table",
    requires={"prices"},
    budget_ms=120_000,
)
def carver_diagnostics(inputs: AnalysisInputs, params: CarverParams) -> AnalysisOutput:
    close = _prices(inputs)["adj_close"].astype(float)
    if len(close) < carver.WARMUP + 504:
        raise InsufficientDataError(
            f"The Carver engine needs ≥ {carver.WARMUP + 504} bars, got {len(close)}"
        )
    periods = carver.bars_per_year(close.index)
    f, _ = carver.forecasts(close, periods)
    fc, fdm = carver.combined(f)
    live = f.dropna()
    costs = bt.Costs(params.fee_bps, params.slippage_bps)
    warmup = carver.WARMUP
    targets = {
        "Carver engine": carver.position(close, params.vol_target, params.buffer),
        "Vol-targeted buy & hold (no trend)": carver.vol_targeted_hold(close, params.vol_target),
        "Buy & hold": pd.Series(1.0, index=close.index),
    }
    frames = {k: bt.run_vectorised(close, t, costs, warmup) for k, t in targets.items()}
    cut = pd.Timestamp(
        frames["Buy & hold"].index[int(len(frames["Buy & hold"]) * params.is_fraction)]
    )
    grid = bt.param_grid(bt.DEFAULT_GRIDS["carver_trend"])
    trials = bt.grid_returns("carver_trend", close, grid, costs, warmup)
    oos_trials = trials.loc[trials.index >= cut]
    srs = [float(c.mean() / c.std(ddof=1)) for _, c in oos_trials.items() if c.std(ddof=1) > 0]
    oos = frames["Carver engine"].loc[lambda x: x.index >= cut, "net"].to_numpy()
    corr = live.corr()
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "split": cut.date().isoformat(),
            "bars_per_year": periods,
            "fdm_latest": _f(fdm.iloc[-1]),
            "combined_mean_abs_forecast": _f(fc.dropna().abs().mean()),
            "rules": [
                {
                    "rule": k,
                    "weight": _f(w),
                    "mean_abs_forecast": _f(live[k].abs().mean()),
                    "share_at_cap": _f((live[k].abs() >= carver.FC_CAP - 1e-9).mean()),
                    "correlation_to_ewmac": _f(corr.loc[k, "ewmac"]),
                    "latest_forecast": _f(f[k].iloc[-1]),
                }
                for k, w in zip(carver.RULE_WEIGHTS, carver.weights(), strict=True)
            ],
            "correlations": [
                {"rule": a, **{b: _f(corr.loc[a, b]) for b in corr.columns}} for a in corr.index
            ],
            "comparison": [_row(k, fr, cut, periods) for k, fr in frames.items()],
            "validation": {
                "trials_in_grid": len(grid),
                "dsr": _f(bt.dsr(oos, srs)),
                "psr": _f(bt.psr(oos)),
                "pbo": _f(bt.pbo(trials, 8)["pbo"]) if len(trials) >= 320 else None,
            },
            "notes": [
                "Diversify across assets, not across indicators: on one equity index the engine "
                "is crash insurance, on a basket of uncorrelated assets it carries the Sharpe.",
                "A Sharpe gap smaller than its standard error is not evidence.",
            ],
        },
    )


class ChallengeParams(BaseModel):
    vol_targets: list[float] = Field(
        default_factory=lambda: [0.08, 0.12, 0.16, 0.20, 0.30, 0.40], min_length=1, max_length=10
    )
    paths: int = Field(5000, ge=500, le=25_000)
    block: int = Field(20, ge=5, le=60)
    phase1_target: float = Field(0.10, gt=0, le=0.5)
    phase2_target: float = Field(0.05, gt=0, le=0.5)
    daily_loss_limit: float = Field(0.05, gt=0, le=0.5)
    max_loss_limit: float = Field(0.10, gt=0, le=0.9)
    min_days: int = Field(4, ge=1, le=30)
    horizon_days: int = Field(250, ge=20, le=750)
    seed: int = 1


@analysis(
    id="TR-019",
    version="1.0.0",
    title="Prop-firm challenge Monte Carlo",
    description=(
        "Probability of passing a two-phase funded-account challenge (default FTMO-style: +10 % "
        "then +5 %, 5 % daily and 10 % total loss limits, ≥ 4 trading days) with the strategy's "
        "out-of-sample daily returns, block-bootstrapped (20-day blocks keep losing streaks) and "
        "rescaled to each volatility target, with the median days to funding. Close-to-close: "
        "real challenges check intraday losses, so the probabilities are slightly optimistic."
    ),
    applies_to={"strategy"},
    params=ChallengeParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def prop_challenge(inputs: AnalysisInputs, params: ChallengeParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    validation = _validation(strat, ev, close)
    frame = bt.run_vectorised(
        close, _target(strat["spec"], close, ev["params"]), ev["costs"], ev["warmup"]
    )
    daily = frame.loc[frame.index >= ev["cut"], "net"].to_numpy()
    if len(daily) < 252:
        raise InsufficientDataError(f"Needs ≥ 252 OOS days, got {len(daily)}")
    rows = carver.prop_challenge(
        daily,
        tuple(params.vol_targets),
        carver.bars_per_year(close.index),
        n_paths=params.paths,
        block=params.block,
        horizon=params.horizon_days,
        targets=(params.phase1_target, params.phase2_target),
        daily_loss=params.daily_loss_limit,
        max_loss=params.max_loss_limit,
        min_days=params.min_days,
        seed=params.seed,
    )
    best = max(rows, key=lambda r: r["p_funded"] or 0.0)
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "validation": validation,
            "oos_days": len(daily),
            "oos_volatility": _f(
                np.std(daily, ddof=1) * math.sqrt(carver.bars_per_year(close.index))
            ),
            "rows": rows,
            "best_vol_target": best["vol_target"],
            "note": (
                "Raising the volatility target passes faster but less often; the review found the "
                "sweet spot near 12–16 % on a low-drawdown, diversified strategy."
            ),
        },
    )
