"""Strategy library, signal monitor & execution costs (ANALYTICS_CATALOG TR-010, TR-011,
TR-013; module M26). Signals use closed bars only (the lake never stores today's bar)."""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core import backtest as bt
from fda_core.analyses.backtesting import LABEL, _strategy, _target
from fda_core.analyses.price import PRICE_SUBJECTS, _prices
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis


class LibraryParams(BaseModel):
    is_fraction: float = Field(0.6, ge=0.3, le=0.9)
    fee_bps: float = Field(5.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0)


@analysis(
    id="TR-010",
    version="1.3.0",
    title="Strategy library report",
    description=(
        "Every single-asset template of the library on the subject with its default grid (warm-up "
        "220, 1-bar shift, costs on, IS/OOS split): rules, family, known failure regimes, IS-best "
        "configuration, its OOS Sharpe, DSR over the grid, PBO, OOS max drawdown and trades — the "
        "validation report each template ships with."
    ),
    applies_to=PRICE_SUBJECTS,
    params=LibraryParams,
    result_kind="table",
    requires={"prices"},
    budget_ms=120_000,
)
def library_report(inputs: AnalysisInputs, params: LibraryParams) -> AnalysisOutput:
    prices = _prices(inputs)
    close = prices["adj_close"].astype(float)
    if len(close) < 220 + 504:
        raise InsufficientDataError(
            f"The library report needs ≥ {220 + 504} bars, got {len(close)}"
        )
    costs = bt.Costs(params.fee_bps, params.slippage_bps)
    rows = []
    for template, ranges in bt.DEFAULT_GRIDS.items():
        grid = bt.param_grid(ranges, bt.ordered_constraint(template))  # type: ignore[arg-type]
        trials = bt.grid_returns(template, close, grid, costs, 220)  # type: ignore[arg-type]
        cut = trials.index[int(len(trials) * params.is_fraction)]
        is_part, oos_part = trials.loc[trials.index < cut], trials.loc[trials.index >= cut]
        is_sr = is_part.apply(lambda c: bt.sharpe(c.to_numpy()))
        best = int(is_sr.idxmax()) if is_sr.notna().any() else 0
        oos = oos_part[best].to_numpy()
        srs = [float(c.mean() / c.std(ddof=1)) for _, c in oos_part.items() if c.std(ddof=1) > 0]
        nav = np.cumprod(1 + oos)
        best_frame = bt.run_vectorised(close, bt.signal(template, close, grid[best]), costs, 220)  # type: ignore[arg-type]
        trades = bt.trades_of(best_frame.loc[best_frame.index >= cut])
        pbo = bt.pbo(trials, 8)["pbo"] if trials.shape[1] >= 2 else None
        doc = bt.TEMPLATE_DOCS[template]
        rows.append(
            {
                "template": template,
                "family": doc["family"],
                "rules": doc["rules"],
                "fails_in": doc["fails"],
                "configurations": len(grid),
                "best_is": grid[best],
                "is_sharpe": _f(is_sr.iloc[best]),
                "oos_sharpe": _f(bt.sharpe(oos)),
                "dsr": _f(bt.dsr(oos, srs)),
                "pbo": _f(pbo),
                "oos_max_drawdown": _f(float((nav / np.maximum.accumulate(nav) - 1).min())),
                "oos_trades": len(trades),
            }
        )
    bh = close.loc[close.index >= cut].pct_change().dropna()
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "split": pd.Timestamp(cut).date().isoformat(),
            "buy_and_hold_oos_sharpe": _f(bt.sharpe(bh.to_numpy())),
            "rows": sorted(rows, key=lambda r: -(r["oos_sharpe"] or -9)),
        },
    )


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── TR-011 signal monitor ────────────────────────────────────────────────────


class MonitorParams(BaseModel):
    history: int = Field(10, ge=1, le=100)


def _distance(spec: dict[str, Any], close: pd.Series, params: dict[str, float]) -> float | None:
    """How far the trigger is, in % of price (crosses: fast − slow average)."""
    t = spec["template"]
    if t in ("ema_cross", "tema_cross", "triple_ema"):
        return float(
            (
                bt.ema(close, int(params["fast"])).iloc[-1]
                - bt.ema(close, int(params["slow"])).iloc[-1]
            )
            / close.iloc[-1]
        )
    if t == "sma_cross":
        fast, slow = (
            close.rolling(int(params["fast"])).mean(),
            close.rolling(int(params["slow"])).mean(),
        )
        return float((fast.iloc[-1] - slow.iloc[-1]) / close.iloc[-1])
    if t == "donchian_breakout":
        n = int(params["window"])
        return float(close.iloc[-1] / close.iloc[-n - 1 : -1].max() - 1)
    return None


@analysis(
    id="TR-011",
    version="1.0.0",
    title="Signal monitor",
    description=(
        "Target position of the strategy at the last closed bar (template + overlays, before the "
        "one-bar shift: the position to hold from the next bar), the action implied versus the "
        "previous bar (buy / sell / hold), the date of the last change, the distance to the trigger"
        " and the recent signal history."
    ),
    applies_to={"strategy"},
    params=MonitorParams,
    result_kind="table",
    requires={"strategy", "prices"},
)
def signal_monitor(inputs: AnalysisInputs, params: MonitorParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    spec = strat["spec"]
    close = prices["adj_close"].astype(float)
    p = {k: float(v) for k, v in spec["params"].items()}
    target = _target(spec, close, p)
    changes = target.diff().fillna(target)
    flips = changes[changes != 0]
    position, previous = float(target.iloc[-1]), float(target.iloc[-2])
    action = "buy" if position > previous else "sell" if position < previous else "hold"
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "symbol": spec["symbol"],
            "as_of": close.index[-1].date().isoformat(),
            "close": float(prices["close"].iloc[-1]),
            "position": position,
            "previous_position": previous,
            "action": action,
            "changed_today": action != "hold",
            "last_change": flips.index[-1].date().isoformat() if len(flips) else None,
            "distance_to_trigger": _distance(spec, close, p),
            "params": p,
            "overlays": spec.get("overlays") or [],
            "rows": [
                {
                    "date": d.date().isoformat(),
                    "position": float(target.loc[d]),
                    "close": float(prices["close"].loc[d]),
                }
                for d in flips.index[-params.history :][::-1]
            ],
        },
    )


# ── TR-013 execution cost models ─────────────────────────────────────────────


class ExecutionParams(BaseModel):
    shares: float = Field(10_000, gt=0)
    days: int = Field(1, ge=1, le=20, description="Execution horizon (sessions)")
    slices: int = Field(13, ge=2, le=390, description="Child orders per day (30-min buckets = 13)")
    eta: float = Field(0.7, gt=0, le=5, description="Square-root impact coefficient")
    risk_aversion: float = Field(1e-6, ge=0, le=1e-2, description="Almgren–Chriss λ")
    schedule: Literal["almgren_chriss", "twap", "vwap"] = "almgren_chriss"


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily bid–ask spread estimate from two-day high/low ranges (Corwin & Schultz 2012),
    negative estimates set to 0."""
    beta = np.log(high / low) ** 2
    beta = beta + beta.shift(1)
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def almgren_chriss(shares: float, n: int, sigma: float, eta: float, lam: float) -> np.ndarray:
    """Optimal liquidation trajectory x_j = X·sinh(κ(T − t_j))/sinh(κT) with κ² = λσ²/η
    (discrete approximation, temporary impact only); returns shares traded per slice."""
    if lam <= 0:
        return np.full(n, shares / n)
    kappa = math.sqrt(lam * sigma**2 / eta)
    t = np.arange(n + 1)
    holdings = (
        shares * np.sinh(kappa * (n - t)) / math.sinh(kappa * n)
        if kappa * n < 700
        else np.where(t == 0, shares, 0.0)
    )
    return np.asarray(-np.diff(holdings))


VWAP_PROFILE = np.array(
    [0.12, 0.09, 0.07, 0.06, 0.06, 0.055, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09, 0.12]
)


@analysis(
    id="TR-013",
    version="1.0.0",
    title="Execution cost model",
    description=(
        "For an order of Q shares: Corwin–Schultz spread from daily highs/lows (median of 1 year), "
        "square-root impact η·σ·√(Q/ADV) with 20-day ADV and daily σ, Almgren–Chriss optimal "
        "schedule (risk aversion λ, temporary impact), TWAP and a U-shaped VWAP profile (13 "
        "half-hour buckets; intraday volumes not stored); expected cost in bp and in currency."
    ),
    applies_to={"asset"},
    params=ExecutionParams,
    result_kind="table",
    requires={"prices"},
)
def execution_costs(inputs: AnalysisInputs, params: ExecutionParams) -> AnalysisOutput:
    prices = _prices(inputs)
    if (
        not {"high", "low", "volume"} <= set(prices.columns)
        or prices["volume"].iloc[-20:].sum() <= 0
    ):
        raise InsufficientDataError("Needs stored highs, lows and volumes")
    last = prices.iloc[-252:]
    spread = float(corwin_schultz(last["high"].astype(float), last["low"].astype(float)).median())
    adv = float(last["volume"].iloc[-20:].mean())
    price = float(prices["close"].iloc[-1])
    sigma = float(np.log(prices["close"].astype(float)).diff().iloc[-252:].std())
    participation = params.shares / (adv * params.days)
    impact = params.eta * sigma * math.sqrt(params.shares / adv)
    n = params.slices * params.days
    if params.schedule == "twap":
        child = np.full(n, params.shares / n)
    elif params.schedule == "vwap":
        profile = np.tile(
            VWAP_PROFILE if params.slices == 13 else np.full(params.slices, 1 / params.slices),
            params.days,
        )
        child = params.shares * profile / profile.sum()
    else:
        child = almgren_chriss(
            params.shares,
            n,
            sigma * price / math.sqrt(params.slices),
            params.eta * sigma * price / math.sqrt(adv),
            params.risk_aversion,
        )
    cost_bp = (spread / 2 + impact) * 10_000
    return AnalysisOutput(
        "table",
        {
            "price": price,
            "shares": params.shares,
            "notional": params.shares * price,
            "adv_20d": adv,
            "participation": participation,
            "daily_volatility": sigma,
            "spread_estimate": spread,
            "impact": impact,
            "expected_cost_bp": cost_bp,
            "expected_cost": cost_bp / 10_000 * params.shares * price,
            "schedule": params.schedule,
            "warning": "Participation above 10 % of ADV" if participation > 0.1 else None,
            "rows": [
                {"slice": i + 1, "shares": float(q), "share_of_order": float(q / params.shares)}
                for i, q in enumerate(child)
            ],
        },
    )
