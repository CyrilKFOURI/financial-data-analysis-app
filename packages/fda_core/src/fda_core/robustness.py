"""Robustness protocol of a strategy (TR-024; Quant Rick's corrected research protocol, R-QT).

Pure NumPy/pandas. Every choice here is made on in-sample data only: plateau selection reads the
in-sample Sharpe of each grid cell and its neighbours; out-of-sample figures are reported beside
it for reading, never used to choose. Benchmarks and variants run through the same engine as the
strategy (`backtest.run_vectorised`: one-bar shift, warm-up, the same costs).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fda_core import backtest as bt
from fda_core import carver

MIN_TRADES_PER_YEAR = 2.0
MIN_EPOCH_YEARS = 1.0


def bars_per_year(index: pd.Index) -> float:
    """252 for exchange calendars, about 365 for crypto: read from the dates, not a constant."""
    return carver.bars_per_year(index)


def _axis_steps(ranges: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    return {k: sorted(float(v) for v in values) for k, values in ranges.items()}


def neighbours(
    grid: Sequence[Mapping[str, float]], i: int, ranges: Mapping[str, Sequence[float]]
) -> list[int]:
    """Cells at most one step away on every axis (the cell itself excluded)."""
    axes = _axis_steps(ranges)
    out = []
    for j, g in enumerate(grid):
        if j == i:
            continue
        if all(
            abs(axes[k].index(float(g[k])) - axes[k].index(float(grid[i][k]))) <= 1 for k in axes
        ):
            out.append(j)
    return out


def on_edge(cell: Mapping[str, float], ranges: Mapping[str, Sequence[float]]) -> bool:
    """A parameter sits at the end of its range (only for axes with three values or more)."""
    axes = _axis_steps(ranges)
    return any(len(v) >= 3 and float(cell[k]) in (v[0], v[-1]) for k, v in axes.items())


def plateau_choice(
    grid: Sequence[Mapping[str, float]],
    is_sharpe: Sequence[float],
    trades_per_year: Sequence[float],
    ranges: Mapping[str, Sequence[float]],
    min_trades: float = MIN_TRADES_PER_YEAR,
) -> dict[str, Any]:
    """Score = median in-sample Sharpe of the cell and its neighbours. Eligible = at least
    `min_trades` a year, and not on the grid's edge when an inner cell exists. Returns the plateau
    pick, the in-sample best and every cell's score."""
    n = len(grid)
    sr = np.asarray(is_sharpe, dtype=float)
    scores = []
    for i in range(n):
        around = [sr[i]] + [sr[j] for j in neighbours(grid, i, ranges)]
        finite = [x for x in around if math.isfinite(x)]
        scores.append(float(np.median(finite)) if finite else math.nan)
    edge = [on_edge(g, ranges) for g in grid]
    active = [float(t) >= min_trades for t in trades_per_year]
    inner_exists = any(not e and a for e, a in zip(edge, active, strict=True))
    eligible = [a and (not e or not inner_exists) for e, a in zip(edge, active, strict=True)]
    candidates = [i for i in range(n) if eligible[i] and math.isfinite(scores[i])]
    finite_is = [i for i in range(n) if math.isfinite(sr[i])]
    return {
        # ties go to the cell with the better own Sharpe
        "pick": max(candidates, key=lambda i: (scores[i], sr[i])) if candidates else None,
        "is_best": max(finite_is, key=lambda i: sr[i]) if finite_is else None,
        "scores": scores,
        "edge": edge,
        "eligible": eligible,
    }


def trades_per_year(frame: pd.DataFrame, periods: float) -> float:
    """Round trips per year (a trade lasts while the position keeps its sign)."""
    years = len(frame) / periods
    return float(len(bt.trades_of(frame)) / years) if years > 0 else math.nan


def sharpe_se(sr: float, years: float) -> float | None:
    """Standard error of an annual Sharpe ratio: √((1 + SR²/2) / T), T in years."""
    if years <= 0 or not math.isfinite(sr):
        return None
    return math.sqrt((1 + sr * sr / 2) / years)


def summary(net: pd.Series, periods: float) -> dict[str, float | None]:
    """Sharpe (± its standard error), CAGR and maximum drawdown of a net-return series."""
    years = len(net) / periods
    nav = (1 + net).cumprod()
    sr = bt.sharpe(net.to_numpy(), round(periods))
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 and nav.iloc[-1] > 0 else None
    return {
        "years": years,
        "sharpe": sr if math.isfinite(sr) else None,
        "sharpe_se": sharpe_se(sr, years),
        "cagr": cagr,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
    }


def benchmark_targets(close: pd.Series, vol_target: float) -> dict[str, pd.Series]:
    """Positions decided at the close (the engine shifts them): buy & hold, the vol-targeted buy &
    hold (Carver's no-trend benchmark, point-in-time) and the MA200 trend filter."""
    ma = close.rolling(200, min_periods=200).mean()
    return {
        "buy_and_hold": pd.Series(1.0, index=close.index),
        "vol_targeted_hold": carver.vol_targeted_hold(close, vol_target),
        "ma200": (close > ma).astype(float).where(ma.notna(), 0.0),
    }


def lag_sweep(
    close: pd.Series,
    target: pd.Series,
    costs: bt.Costs,
    warmup: int,
    periods: float,
    adj_open: pd.Series | None = None,
    lags: Sequence[int] = (1, 2, 3),
) -> list[dict[str, Any]]:
    """The same decisions executed later. Decided at close t: lag 1 trades at that close (the
    engine's default), lag 2 at close t+1 (the TEMA notebook's timing), lag 3 at close t+2; and
    at the open of t+1 = open-to-open returns with a two-bar shift."""
    rows = []
    for lag in lags:
        frame = bt.run_vectorised(close, target, costs, warmup, lag)
        label = "close t" if lag == 1 else f"close t+{lag - 1}"
        rows.append({"execution": label, "lag": lag, **summary(frame["net"], periods)})
    if adj_open is not None and adj_open.notna().all():
        frame = bt.run_vectorised(adj_open, target, costs, warmup, 2)
        rows.append({"execution": "open t+1", "lag": None, **summary(frame["net"], periods)})
    return rows


def epochs(
    frames: Mapping[str, pd.DataFrame], split: pd.Timestamp, periods: float
) -> list[dict[str, Any]]:
    """Each series summarised before and after `split` (an epoch shorter than a year gives null
    figures and the reason)."""
    rows = []
    for name, frame in frames.items():
        for label, part in (
            ("epoch 1", frame.loc[frame.index < split]),
            ("epoch 2", frame.loc[frame.index >= split]),
        ):
            years = len(part) / periods
            if years < MIN_EPOCH_YEARS:
                rows.append(
                    {
                        "series": name,
                        "epoch": label,
                        "years": years,
                        "sharpe": None,
                        "sharpe_se": None,
                        "cagr": None,
                        "max_drawdown": None,
                        "reason": f"{years:.1f} years: under the one-year minimum",
                    }
                )
                continue
            rows.append(
                {
                    "series": name,
                    "epoch": label,
                    "start": part.index[0].date().isoformat(),
                    "end": part.index[-1].date().isoformat(),
                    **summary(part["net"], periods),
                }
            )
    return rows


def epoch_sharpes(rows: Sequence[Mapping[str, Any]], series: str) -> list[float | None]:
    return [r["sharpe"] for r in rows if r["series"] == series]


def ablation(
    variants: Mapping[str, pd.DataFrame],
    full: pd.DataFrame,
    split: pd.Timestamp,
    periods: float,
) -> list[dict[str, Any]]:
    """Each layer switched off: Sharpe in both epochs with and without it. A layer earns its place
    when removing it lowers the Sharpe in both epochs."""
    base = epochs({"full": full}, split, periods)
    s1, s2 = (r["sharpe"] for r in base)
    rows = []
    for layer, frame in variants.items():
        e = epochs({layer: frame}, split, periods)
        w1, w2 = (r["sharpe"] for r in e)
        known = s1 is not None and s2 is not None and w1 is not None and w2 is not None
        rows.append(
            {
                "layer": layer,
                "sharpe_epoch_1_without": w1,
                "sharpe_epoch_2_without": w2,
                "delta_epoch_1": (s1 - w1) if known else None,
                "delta_epoch_2": (s2 - w2) if known else None,
                "earns_its_place": bool(known and s1 > w1 and s2 > w2),
            }
        )
    return rows


def check(name: str, ok: bool | None, detail: str) -> dict[str, Any]:
    """One line of the verdict: pass, warn, or n/a when it cannot be computed."""
    return {
        "check": name,
        "status": "n/a" if ok is None else "pass" if ok else "warn",
        "detail": detail,
    }


# ── ensembles and the shadow test (TR-025) ───────────────────────────────────

ENSEMBLE_RULES = ("or", "and", "vote", "mean")


def combine(signals: pd.DataFrame, rule: str) -> pd.Series:
    """Members decided at the close, combined: OR (any long), AND (all long), majority vote
    (more than half long), mean (average position, keeps continuous sizes)."""
    long = signals > 0
    if rule == "or":
        out = long.any(axis=1).astype(float)
    elif rule == "and":
        out = long.all(axis=1).astype(float)
    elif rule == "vote":
        out = (long.mean(axis=1) > 0.5).astype(float)
    elif rule == "mean":
        out = signals.mean(axis=1)
    else:
        raise ValueError(f"Unknown ensemble rule {rule!r}")
    return out.astype(float)


def shadow_test(
    frame: pd.DataFrame,
    costs: bt.Costs,
    periods: float,
    n: int = 200,
    seed: int = 1,
    min_offset: int = 63,
) -> dict[str, Any]:
    """Does the signal beat its own shadows? A shadow is the same position path rotated in time
    by a random offset (at least `min_offset` bars from both ends): same exposure, same trade
    rhythm, timing destroyed. Returns the Sharpe, the shadows' 95th percentile and
    p = (1 + #shadows ≥ actual) / (n + 1). Meant for in-sample bars only."""
    position = frame["position"].to_numpy()
    r = frame["asset"].to_numpy()
    m = len(position)
    ann = round(periods)
    actual = bt.sharpe(frame["net"].to_numpy(), ann)
    if m < 2 * min_offset + 2 or not math.isfinite(actual):
        return {
            "sharpe": actual if math.isfinite(actual) else None,
            "p_value": None,
            "shadow_p95": None,
        }
    rng = np.random.default_rng(seed)
    offsets = rng.integers(min_offset, m - min_offset, size=n)
    shadows = []
    for k in offsets:
        p = np.roll(position, int(k))
        traded = np.abs(np.diff(p, prepend=0.0))
        shadows.append(bt.sharpe(p * r - traded * costs.per_unit, ann))
    s = np.asarray([x for x in shadows if math.isfinite(x)])
    return {
        "sharpe": actual,
        "shadow_p95": float(np.quantile(s, 0.95)) if len(s) else None,
        "p_value": float((1 + (s >= actual).sum()) / (len(s) + 1)) if len(s) else None,
    }


# ── leveraged ETFs and the LETF baseline (311c) ──────────────────────────────


def leveraged(
    close: pd.Series, factor: float, fee: float, financing: float, periods: float
) -> pd.Series:
    """A daily-reset ×factor ETF simulated from its index (for histories before the ETF):
    r_k = factor·r − (fee + (factor − 1)·financing) / periods, priced from 100. `fee` is the
    annual expense ratio and `financing` the annual cost of the borrowed exposure."""
    r = close.pct_change().fillna(0.0)
    drag = (fee + (factor - 1) * financing) / periods
    growth = (1 + factor * r - drag).clip(lower=0.0)
    growth.iloc[0] = 1.0  # the first bar is the starting price, not a day held
    return 100 * growth.cumprod()


def letf_baseline(
    close: pd.Series,
    costs: bt.Costs,
    warmup: int,
    periods: float,
    factor: float = 3.0,
    fee: float = 0.0088,
    financing: float = 0.04,
) -> pd.DataFrame:
    """The baseline Rick accepted for leveraged trend systems (Community, 2026-10-13): 50/50 of
    the index bought and held and the ×factor ETF held only above its own 200-day average,
    rebalanced daily. Both sleeves run through the engine (one-bar shift, the same costs)."""
    lev = leveraged(close, factor, fee, financing, periods)
    ma = lev.rolling(200, min_periods=200).mean()
    hold = bt.run_vectorised(close, pd.Series(1.0, index=close.index), costs, warmup)
    trend = bt.run_vectorised(lev, (lev > ma).astype(float).where(ma.notna(), 0.0), costs, warmup)
    return pd.DataFrame(
        {
            "position": 0.5 * hold["position"] + 0.5 * factor * trend["position"],
            "gross": 0.5 * (hold["gross"] + trend["gross"]),
            "cost": 0.5 * (hold["cost"] + trend["cost"]),
            "net": 0.5 * (hold["net"] + trend["net"]),
            "asset": hold["asset"],
        }
    )


# ── CFD costs (prop firms, 311c) ─────────────────────────────────────────────


def cfd_net(
    frame: pd.DataFrame, spread_bps: float, swap_long: float, swap_short: float
) -> pd.Series:
    """Net returns of the same positions held as CFDs: the spread is paid on each change of
    position, the swap every night held (annual rates, positive = cost, charged per calendar
    night, so a Friday-to-Monday hold pays three nights; the broker's triple-swap day moves the
    charge inside the week, not its weekly total). No exchange fees on a CFD."""
    position = frame["position"]
    nights = frame.index.to_series().diff().dt.days.fillna(1).clip(lower=1)
    rate = np.where(position > 0, swap_long, np.where(position < 0, swap_short, 0.0))
    swap = position.abs() * rate * nights / 365
    spread = position.diff().abs().fillna(position.abs()) * spread_bps / 10_000
    return frame["position"] * frame["asset"] - spread - swap
