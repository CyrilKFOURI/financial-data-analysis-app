"""Backtesting engine & validation (M25, TR-001…TR-009, TR-014…TR-017): pure NumPy/pandas.

Doctrine (AFML-11): signals are shifted one bar (decided on bar t's close, held over t+1);
indicators are warmed up and only evaluated after `warmup` bars; costs (fees + slippage, bps of
traded notional) are always on; every configuration run counts as a trial for DSR and PBO.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from fda_core import carver
from fda_core.registry import InsufficientDataError

Template = Literal[
    "ema_cross",
    "triple_ema",
    "sma_cross",
    "rsi_reversion",
    "bollinger_reversion",
    "donchian_breakout",
    "momentum",
    "zscore_reversion",
    "tema_cross",
    "supertrend",
    "short_term_reversal",
    "carver_trend",
    "macd_v",
    "donchian_ensemble",
]

# Default grids (the library's reference configuration space, TR-010).
DEFAULT_GRIDS: dict[str, dict[str, list[float]]] = {
    "ema_cross": {"fast": [5, 10, 20, 30], "slow": [50, 100, 150, 200]},
    "triple_ema": {"fast": [5, 7, 10], "mid": [20, 30, 50], "slow": [93, 150, 200]},
    "sma_cross": {"fast": [10, 20, 50], "slow": [100, 150, 200]},
    "tema_cross": {"fast": [10, 20, 30], "slow": [50, 100, 150]},
    "rsi_reversion": {"period": [2, 7, 14], "lower": [10, 20, 30], "upper": [60, 70]},
    "bollinger_reversion": {"window": [10, 20, 40], "k": [1.5, 2.0, 2.5]},
    "donchian_breakout": {"window": [20, 55, 100]},
    "supertrend": {"period": [7, 10, 14], "multiplier": [2.0, 3.0, 4.0]},
    "momentum": {"lookback": [126, 252], "skip": [0, 21]},
    "zscore_reversion": {"window": [20, 60, 120], "entry": [1.5, 2.0, 2.5]},
    "short_term_reversal": {"lookback": [3, 5, 10], "threshold": [0.02, 0.04]},
    "carver_trend": {"vol_target": [0.12, 0.16, 0.2], "buffer": [0.0, 0.1]},
    "macd_v": {"fast": [8, 12], "slow": [21, 26], "threshold": [0, 50]},
    "donchian_ensemble": {"shortest": [10, 20, 40], "levels": [3, 4]},
}

# Library documentation (TR-010): family, rules and known failure regimes.
TEMPLATE_DOCS: dict[str, dict[str, str]] = {
    "ema_cross": {
        "family": "trend",
        "rules": "Long when EMA(fast) > EMA(slow)",
        "fails": "Choppy, range-bound markets (whipsaws)",
    },
    "sma_cross": {
        "family": "trend",
        "rules": "Long when SMA(fast) > SMA(slow)",
        "fails": "Whipsaws; slow to exit V-shaped crashes",
    },
    "triple_ema": {
        "family": "trend",
        "rules": "Long when EMA(fast) > EMA(mid) > EMA(slow)",
        "fails": "Late entries after sharp reversals",
    },
    "tema_cross": {
        "family": "trend",
        "rules": "Long when TEMA(fast) > TEMA(slow); TEMA = 3E1 − 3E2 + E3",
        "fails": "Over-reacts to noise at short spans",
    },
    "donchian_breakout": {
        "family": "trend",
        "rules": "Long above the prior N-day high, flat below the prior low",
        "fails": "False breakouts in ranges",
    },
    "supertrend": {
        "family": "trend",
        "rules": "Long above a trailing close − m·ATR band (close-to-close ATR)",
        "fails": "Gaps through the band",
    },
    "momentum": {
        "family": "trend",
        "rules": "Long when the t−lookback → t−skip return is positive (12-1)",
        "fails": "Momentum crashes at market rebounds",
    },
    "rsi_reversion": {
        "family": "mean reversion",
        "rules": "Long when RSI < lower, exit when RSI > upper",
        "fails": "Persistent trends (falling knives)",
    },
    "bollinger_reversion": {
        "family": "mean reversion",
        "rules": "Long below MA − k·σ, exit at the MA",
        "fails": "Trending regimes with expanding volatility",
    },
    "zscore_reversion": {
        "family": "mean reversion",
        "rules": "Long at z < −entry, short at z > entry, exit near 0",
        "fails": "Structural breaks in the mean",
    },
    "short_term_reversal": {
        "family": "mean reversion",
        "rules": "Long for N days after a fall > threshold over N days",
        "fails": "Crash regimes where losses cluster",
    },
    "carver_trend": {
        "family": "trend",
        "rules": (
            "Carver forecasts (EWMAC, breakout, acceleration, skew) combined with a point-in-time "
            "FDM; weight = F/10 · vol target / σ, long-only, capped at 1, 10 % buffer"
        ),
        "fails": "V-shaped rebounds on a rising asset (lags a vol-targeted buy & hold)",
    },
    "macd_v": {
        "family": "trend",
        "rules": (
            "MACD-V = (EMA fast − EMA slow) / ATR slow × 100 (close-to-close ATR); long while "
            "MACD-V is above its 9-bar signal and above the threshold"
        ),
        "fails": "Low-volatility drifts, where the ATR scaling inflates small crosses",
    },
    "donchian_ensemble": {
        "family": "trend",
        "rules": (
            "Average of Donchian channels N = shortest × 2^k (k < levels): each enters above "
            "its prior N-day high and exits below the channel's middle; position = share long"
        ),
        "fails": "Range-bound markets after a long trend (every channel exits late)",
    },
}


# ── signals (TR-001) ─────────────────────────────────────────────────────────


def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    total = gain + loss
    return (100 * gain / total.where(total > 0)).fillna(50)


def signal(
    template: Template, close: pd.Series, params: dict[str, float], short: bool = False
) -> pd.Series:
    """Target position on each bar's close (before the one-bar shift): 1 long, −1 short, 0 flat."""
    p = {k: int(v) if float(v).is_integer() else v for k, v in params.items()}
    if template == "ema_cross":
        raw = np.sign(ema(close, int(p["fast"])) - ema(close, int(p["slow"])))
    elif template == "sma_cross":
        raw = np.sign(close.rolling(int(p["fast"])).mean() - close.rolling(int(p["slow"])).mean())
    elif template == "triple_ema":
        f, m, s = ema(close, int(p["fast"])), ema(close, int(p["mid"])), ema(close, int(p["slow"]))
        raw = pd.Series(
            np.where((f > m) & (m > s), 1.0, np.where((f < m) & (m < s), -1.0, 0.0)),
            index=close.index,
        )
    elif template == "momentum":
        lookback, skip = int(p.get("lookback", 252)), int(p.get("skip", 21))
        raw = np.sign(close.shift(skip) / close.shift(lookback) - 1)
    elif template == "donchian_breakout":
        n = int(p["window"])
        upper, lower = close.rolling(n).max().shift(1), close.rolling(n).min().shift(1)
        state = pd.Series(
            np.where(close > upper, 1.0, np.where(close < lower, -1.0, np.nan)), index=close.index
        )
        raw = state.ffill().fillna(0.0)
    elif template == "rsi_reversion":
        r = rsi(close, int(p.get("period", 14)))
        state = pd.Series(
            np.where(r < p.get("lower", 30), 1.0, np.where(r > p.get("upper", 70), 0.0, np.nan)),
            index=close.index,
        )
        raw = state.ffill().fillna(0.0)
    elif template == "bollinger_reversion":
        n, k = int(p.get("window", 20)), float(p.get("k", 2.0))
        mid, sd = close.rolling(n).mean(), close.rolling(n).std()
        state = pd.Series(
            np.where(close < mid - k * sd, 1.0, np.where(close > mid, 0.0, np.nan)),
            index=close.index,
        )
        raw = state.ffill().fillna(0.0)
    elif template == "zscore_reversion":
        n, entry = int(p.get("window", 60)), float(p.get("entry", 2.0))
        z = (close - close.rolling(n).mean()) / close.rolling(n).std()
        state = pd.Series(
            np.where(
                z < -entry, 1.0, np.where(z > entry, -1.0, np.where(z.abs() < 0.5, 0.0, np.nan))
            ),
            index=close.index,
        )
        raw = state.ffill().fillna(0.0)
    elif template == "tema_cross":
        raw = np.sign(_tema(close, int(p["fast"])) - _tema(close, int(p["slow"])))
    elif template == "supertrend":
        raw = _supertrend(close, int(p.get("period", 10)), float(p.get("multiplier", 3.0)))
    elif template == "carver_trend":
        return carver.position(
            close, float(p.get("vol_target", 0.2)), float(p.get("buffer", 0.1)), not short
        )
    elif template == "macd_v":
        return _macd_v(close, int(p["fast"]), int(p["slow"]), float(p.get("threshold", 0.0)))
    elif template == "donchian_ensemble":
        return _donchian_ensemble(close, int(p["shortest"]), int(p["levels"]))
    elif template == "short_term_reversal":
        n, threshold = int(p.get("lookback", 5)), float(p.get("threshold", 0.03))
        drop = (close / close.shift(n) - 1) < -threshold
        raw = drop.astype(float).rolling(n, min_periods=1).max()
    else:
        raise InsufficientDataError(f"Unknown template {template!r}")
    out = pd.Series(raw, index=close.index).fillna(0.0).astype(float)
    return out if short else out.clip(lower=0.0)


def _tema(x: pd.Series, n: int) -> pd.Series:
    e1 = ema(x, n)
    e2 = ema(e1, n)
    return 3 * e1 - 3 * e2 + ema(e2, n)


def macd_v(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """MACD-V (Spiroglou): MACD in units of volatility, (EMA fast − EMA slow) / ATR slow × 100.
    The ATR is close-to-close (Wilder smoothing of |Δclose|), like the supertrend template."""
    atr = close.diff().abs().ewm(alpha=1 / slow, adjust=False, min_periods=slow).mean()
    return (ema(close, fast) - ema(close, slow)) / atr * 100


MAX_DONCHIAN_LEVELS = 6


def _macd_v(close: pd.Series, fast: int, slow: int, threshold: float) -> pd.Series:
    if not 2 <= fast < slow <= len(close):
        raise InsufficientDataError("macd_v needs 2 ≤ fast < slow ≤ the number of bars")
    value = macd_v(close, fast, slow)
    sig = value.ewm(span=9, adjust=False).mean()
    return ((value > sig) & (value > threshold)).astype(float)


def _donchian_ensemble(close: pd.Series, shortest: int, levels: int) -> pd.Series:
    """Share of the channels N = shortest·2^k that are long. Each channel enters above its prior
    N-day high and exits below the middle of its prior N-day range (long-only). Parameters come
    from user grids, so they are bounded: shortest ≥ 2, 1 ≤ levels ≤ 6, longest ≤ the series."""
    if shortest < 2 or not 1 <= levels <= MAX_DONCHIAN_LEVELS:
        raise InsufficientDataError(
            f"donchian_ensemble needs shortest ≥ 2 and 1 ≤ levels ≤ {MAX_DONCHIAN_LEVELS}"
        )
    if shortest * 2 ** (levels - 1) > len(close):
        raise InsufficientDataError(
            f"The longest channel ({shortest * 2 ** (levels - 1)} bars) exceeds the series"
        )
    states = []
    for k in range(levels):
        n = shortest * 2**k
        high = close.rolling(n).max().shift(1)
        low = close.rolling(n).min().shift(1)
        mid = (high + low) / 2
        state = pd.Series(
            np.where(close > high, 1.0, np.where(close < mid, 0.0, np.nan)), index=close.index
        )
        states.append(state.ffill().fillna(0.0))
    return pd.concat(states, axis=1).mean(axis=1)


def _supertrend(close: pd.Series, n: int, mult: float) -> pd.Series:
    """Trailing band flip on closes: direction changes when the close crosses the band."""
    atr = close.diff().abs().ewm(alpha=1 / n, adjust=False).mean()
    lower = (close - mult * atr).to_numpy()
    upper = (close + mult * atr).to_numpy()
    c = close.to_numpy()
    state = np.zeros(len(c))
    trail_low, trail_high, direction = lower[0], upper[0], 1.0
    for i in range(1, len(c)):
        if direction > 0:
            trail_low = max(trail_low, lower[i])
            if c[i] < trail_low:
                direction, trail_high = -1.0, upper[i]
        else:
            trail_high = min(trail_high, upper[i])
            if c[i] > trail_high:
                direction, trail_low = 1.0, lower[i]
        state[i] = direction
    return pd.Series(state, index=close.index)


def apply_overlays(
    target: pd.Series,
    close: pd.Series,
    overlays: list[str],
    vol_target: float = 0.15,
    max_leverage: float = 1.5,
    drawdown_limit: float = 0.15,
) -> pd.Series:
    """Overlays on the target, still decided at the close (the engine shifts later): trend
    filter = flat below the 200-day average; vol targeting = × target / 20-day realised vol
    (capped); drawdown control = flat while the asset is more than the limit below its peak."""
    out = target.copy()
    if "trend_filter" in overlays:
        out = out.where(close >= close.rolling(200, min_periods=200).mean(), 0.0)
    if "vol_target" in overlays:
        realised = close.pct_change().rolling(20).std() * math.sqrt(252)
        out = out * (vol_target / realised).clip(upper=max_leverage).fillna(0.0)
    if "drawdown_control" in overlays:
        dd = close / close.cummax() - 1
        out = out.where(dd > -drawdown_limit, 0.0)
    return out.fillna(0.0)


# ── vectorised engine (TR-001) ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Costs:
    fee_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def per_unit(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000


def run_vectorised(
    close: pd.Series, target: pd.Series, costs: Costs, warmup: int = 220, lag: int = 1
) -> pd.DataFrame:
    """position_t = target_{t−lag}; net_t = position_t·r_t − costs·|position_t − position_{t−1}|.
    Bars before `warmup` are excluded (positions forced flat, no trade at the boundary)."""
    r = close.pct_change().fillna(0.0)
    position = target.shift(lag).fillna(0.0)
    position.iloc[:warmup] = 0.0
    trades = position.diff().abs().fillna(position.abs())
    gross = position * r
    net = gross - trades * costs.per_unit
    frame = pd.DataFrame(
        {
            "position": position,
            "gross": gross,
            "cost": trades * costs.per_unit,
            "net": net,
            "asset": r,
        }
    )
    return frame.iloc[warmup:]


# ── statistics (TR-005, TR-006, TR-016) ──────────────────────────────────────


def sharpe(r: np.ndarray | pd.Series, periods: int = 252) -> float:
    x = np.asarray(r, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(periods)) if sd > 0 else math.nan


def trades_of(frame: pd.DataFrame) -> pd.DataFrame:
    """Round trips from the position path (a trade lasts while the position keeps its sign:
    resizing a continuous position is not a new trade): entry/exit bars, direction, compounded
    net return (the exit bar's cost is charged to the closing trade, pro rata of the units it
    closes), duration, MAE/MFE on close-to-close marks."""
    pos = frame["position"].to_numpy()
    net = frame["net"].to_numpy().copy()
    cost = frame["cost"].to_numpy()
    idx = frame.index
    rows: list[dict[str, Any]] = []
    start: int | None = None
    for i in range(len(pos)):
        prev = pos[i - 1] if i > 0 else 0.0
        if np.sign(pos[i]) != np.sign(prev):
            if start is not None:
                share = abs(prev) / abs(pos[i] - prev)
                exit_cost = cost[i] * share
                net[i] += exit_cost  # the new position keeps only its own entry cost
                seg = np.append(net[start:i], -exit_cost)
                path = np.cumprod(1 + seg) - 1
                rows.append(
                    {
                        "entry": idx[start],
                        "exit": idx[i - 1],
                        "side": float(np.sign(pos[start])),
                        "return": float(np.prod(1 + seg) - 1),
                        "bars": i - start,
                        "mae": float(min(path.min(), 0)),
                        "mfe": float(max(path.max(), 0)),
                    }
                )
                start = None
            if pos[i] != 0:
                start = i
    if start is not None:
        seg = net[start:]
        path = np.cumprod(1 + seg) - 1
        rows.append(
            {
                "entry": idx[start],
                "exit": idx[-1],
                "side": float(np.sign(pos[start])),
                "return": float(np.prod(1 + seg) - 1),
                "bars": len(pos) - start,
                "mae": float(min(path.min(), 0)),
                "mfe": float(max(path.max(), 0)),
                "open": True,
            }
        )
    return pd.DataFrame(rows)


def _streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def trade_metrics(trades: pd.DataFrame) -> dict[str, float | None]:
    """TEMA-notebook trade analytics (TradingView-style)."""
    if trades.empty:
        return {"trades": 0}
    r = trades["return"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    expectancy = float(r.mean())
    sqn = (
        float(math.sqrt(len(r)) * r.mean() / r.std(ddof=1))
        if len(r) > 1 and r.std(ddof=1) > 0
        else None
    )
    return {
        "trades": len(r),
        "win_rate": float(len(wins) / len(r)),
        "expectancy": expectancy,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "payoff_ratio": float(wins.mean() / -losses.mean())
        if len(wins) and len(losses) and losses.mean() < 0
        else None,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "largest_win": float(r.max()),
        "largest_loss": float(r.min()),
        "sqn": sqn,
        "max_win_streak": _streak(r > 0),
        "max_loss_streak": _streak(r <= 0),
        "avg_bars": float(trades["bars"].mean()),
        "avg_bars_win": float(trades.loc[r > 0, "bars"].mean()) if len(wins) else None,
        "avg_mae": float(trades["mae"].mean()),
        "avg_mfe": float(trades["mfe"].mean()),
        "long_share": float((trades["side"] > 0).mean()),
    }


def performance(frame: pd.DataFrame, periods: int = 252, rf: float = 0.0) -> dict[str, Any]:
    """AFML-14 general characteristics, performance, runs and efficiency on net returns."""
    r = frame["net"]
    nav = (1 + r).cumprod()
    years = len(r) / periods
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 and nav.iloc[-1] > 0 else math.nan
    vol = float(r.std(ddof=1) * math.sqrt(periods))
    dd = nav / nav.cummax() - 1
    downside = r[r < 0].std(ddof=1) * math.sqrt(periods)
    under = (dd < 0).astype(int)
    longest_under = (
        int(under.groupby((under != under.shift()).cumsum()).sum().max()) if len(under) else 0
    )
    pos = r[r > 0]
    neg = r[r < 0]
    hhi_pos = float(((pos / pos.sum()) ** 2).sum()) if len(pos) > 1 and pos.sum() > 0 else None
    q95, q05 = r.quantile(0.95), r.quantile(0.05)
    turnover = float(frame["position"].diff().abs().sum() / years) if years > 0 else math.nan
    exposure = frame["position"].abs()
    return {
        "start": frame.index[0].date().isoformat(),
        "end": frame.index[-1].date().isoformat(),
        "years": years,
        "total_return": float(nav.iloc[-1] - 1),
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe(r - rf / periods, periods),
        "sortino": (cagr - rf) / downside if downside > 0 else None,
        "calmar": cagr / -dd.min() if dd.min() < 0 else None,
        "max_drawdown": float(dd.min()),
        "longest_under_water_bars": longest_under,
        "omega": float(pos.sum() / -neg.sum()) if neg.sum() < 0 else None,
        "tail_ratio": float(q95 / -q05) if q05 < 0 else None,
        "hit_ratio": float((r > 0).sum() / max((r != 0).sum(), 1)),
        "hhi_positive": hhi_pos,
        "exposure": float(exposure.mean()),
        "long_ratio": float((frame["position"] > 0).mean()),
        "turnover_per_year": turnover,
        "costs_total": float(frame["cost"].sum()),
        "correlation_to_asset": float(np.corrcoef(r, frame["asset"])[0, 1])
        if r.std() > 0 and frame["asset"].std() > 0
        else None,
        "buy_and_hold_cagr": float((1 + frame["asset"]).prod() ** (1 / years) - 1)
        if years > 0
        else None,
    }


def psr(r: np.ndarray, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe of per-period returns vs a per-period benchmark SR*."""
    x = np.asarray(r, dtype=float)
    sd = x.std(ddof=1)
    if sd == 0 or len(x) < 3:
        return math.nan
    sr = x.mean() / sd
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(len(x) - 1) / den))


def expected_max_sharpe(trial_srs: list[float]) -> float:
    """SR₀ = √V[SR]·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))) (per-period units)."""
    n = len(trial_srs)
    if n < 2:
        return 0.0
    g = 0.5772156649015329
    return float(
        np.std(trial_srs, ddof=1)
        * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e)))
    )


def dsr(r: np.ndarray, trial_srs: list[float]) -> float:
    return psr(r, expected_max_sharpe(trial_srs))


def haircut_sharpe(sr_annual: float, n_obs: int, trials: int, periods: int = 252) -> float:
    """Harvey–Liu Bonferroni haircut: p-value of the Sharpe t-stat × trials, back to a Sharpe."""
    t = sr_annual * math.sqrt(n_obs / periods)
    p = min(1.0, 2 * stats.norm.sf(abs(t)) * max(trials, 1))
    t_adj = stats.norm.isf(p / 2) if p < 1 else 0.0
    return float(math.copysign(t_adj / math.sqrt(n_obs / periods), sr_annual))


# ── grids, IS/OOS, walk-forward (TR-002, TR-003) ─────────────────────────────


def param_grid(
    ranges: dict[str, list[float]], constraint: Callable[[dict[str, float]], bool] | None = None
) -> list[dict[str, float]]:
    keys = list(ranges)
    out = [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*(ranges[k] for k in keys))
    ]
    return [p for p in out if constraint is None or constraint(p)]


def grid_returns(
    template: Template,
    close: pd.Series,
    grid: list[dict[str, float]],
    costs: Costs,
    warmup: int,
    short: bool = False,
    overlays: list[str] | None = None,
) -> pd.DataFrame:
    """Trial matrix: one column of net returns per configuration (the input of DSR and PBO)."""
    cols = {}
    for i, params in enumerate(grid):
        target = apply_overlays(signal(template, close, params, short), close, overlays or [])
        cols[i] = run_vectorised(close, target, costs, warmup)["net"]
    return pd.DataFrame(cols)


def ordered_constraint(template: Template) -> Callable[[dict[str, float]], bool] | None:
    if template in ("ema_cross", "sma_cross", "tema_cross", "macd_v"):
        return lambda p: p["fast"] < p["slow"]
    if template == "triple_ema":
        return lambda p: p["fast"] < p["mid"] < p["slow"]
    if template == "rsi_reversion":
        return lambda p: p["lower"] < p["upper"]
    return None


def walk_forward(trials: pd.DataFrame, windows: int = 5, backward: bool = False) -> dict[str, Any]:
    """Anchored-free rolling selection: split time into `windows`+1 blocks; in each step pick the
    best-Sharpe configuration on the previous block and record its next-block returns. With
    `backward`, time is reversed (AFML-12 walk-backward consistency)."""
    data = trials.iloc[::-1] if backward else trials
    blocks = np.array_split(np.arange(len(data)), windows + 1)
    oos, chosen = [], []
    for train, test in itertools.pairwise(blocks):
        srs = data.iloc[train].apply(lambda c: sharpe(c.to_numpy()))
        best = int(srs.idxmax())
        chosen.append(best)
        oos.append(data.iloc[test][best])
    path = pd.concat(oos)
    return {
        "sharpe": sharpe(path.to_numpy()),
        "chosen": chosen,
        "returns": path.iloc[::-1] if backward else path,
    }


# ── CSCV / PBO (TR-009) ──────────────────────────────────────────────────────


def pbo(trials: pd.DataFrame, s: int = 16) -> dict[str, Any]:
    """Bailey et al. CSCV: S row blocks; for every half/half combination pick the IS-best
    configuration, find its OOS rank ω̄; logit λ = ln(ω̄/(1−ω̄)); PBO = P(λ ≤ 0). Also the
    OOS-vs-IS Sharpe regression slope and the probability of an OOS loss."""
    if trials.shape[1] < 2:
        raise InsufficientDataError("PBO needs ≥ 2 trials")
    x = trials.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(len(x)), s)
    # Per-block sums, sums of squares and counts: each split aggregates S rows, not T.
    b_sum = np.array([x[b].sum(0) for b in blocks])
    b_sq = np.array([(x[b] ** 2).sum(0) for b in blocks])
    b_n = np.array([len(b) for b in blocks], dtype=float)
    total_sum, total_sq, total_n = b_sum.sum(0), b_sq.sum(0), b_n.sum()
    logits, is_sr, oos_sr = [], [], []

    def ratio(sm: np.ndarray, sq: np.ndarray, n: float) -> np.ndarray:
        mean = sm / n
        var = np.maximum((sq - n * mean**2) / (n - 1), 1e-300)
        return np.asarray(mean / np.sqrt(var))

    for combo in itertools.combinations(range(s), s // 2):
        idx = list(combo)
        is_sum, is_sq, is_n = b_sum[idx].sum(0), b_sq[idx].sum(0), b_n[idx].sum()
        is_perf = ratio(is_sum, is_sq, is_n)
        oos_perf = ratio(total_sum - is_sum, total_sq - is_sq, total_n - is_n)
        best = int(np.nanargmax(is_perf))
        rank = stats.rankdata(oos_perf)[best] / (len(oos_perf) + 1)
        logits.append(math.log(rank / (1 - rank)))
        is_sr.append(is_perf[best])
        oos_sr.append(oos_perf[best])
    logits_arr = np.array(logits)
    slope = float(np.polyfit(is_sr, oos_sr, 1)[0]) if np.std(is_sr) > 0 else math.nan
    return {
        "pbo": float((logits_arr <= 0).mean()),
        "combinations": len(logits),
        "logit_mean": float(logits_arr.mean()),
        "degradation_slope": slope,
        "prob_oos_loss": float((np.array(oos_sr) < 0).mean()),
        "logits": logits_arr.tolist(),
    }


# ── CPCV (TR-014) ────────────────────────────────────────────────────────────


def cpcv_splits(
    n_obs: int, n_groups: int, k_test: int, embargo: int = 0, label_span: int = 0
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    """All C(N, k) train/test splits; training bars within `label_span` before a tested group
    (overlapping labels) and within `embargo` after it are purged — around EACH tested group."""
    groups = np.array_split(np.arange(n_obs), n_groups)
    splits = []
    for combo in itertools.combinations(range(n_groups), k_test):
        test = np.concatenate([groups[g] for g in combo])
        banned = np.zeros(n_obs, dtype=bool)
        for g in combo:
            lo, hi = groups[g][0], groups[g][-1]
            banned[max(0, lo - label_span) : min(n_obs, hi + 1 + embargo)] = True
        train = np.where(~banned)[0]
        splits.append((train, test, combo))
    return splits


def cpcv_paths(
    trials: pd.DataFrame, n_groups: int = 6, k_test: int = 2, embargo: int = 5
) -> dict[str, Any]:
    """φ = k/N·C(N,k) paths: in each split pick the best-Sharpe trial on the training bars and
    collect its test-group returns; group g's returns appear in φ splits, assigned to paths in
    order. Sharpe per path; variance of the mean path Sharpe φ⁻¹σ²(1 + (φ − 1)·avg_corr)."""
    x = trials.to_numpy(dtype=float)
    n = len(x)
    groups = np.array_split(np.arange(n), n_groups)
    splits = cpcv_splits(n, n_groups, k_test, embargo)
    phi = k_test * math.comb(n_groups, k_test) // n_groups
    per_group: dict[int, list[np.ndarray]] = {g: [] for g in range(n_groups)}
    for train, _, combo in splits:
        srs = x[train].mean(0) / x[train].std(0, ddof=1)
        best = int(np.nanargmax(srs))
        for g in combo:
            per_group[g].append(x[groups[g], best])
    paths = [np.concatenate([per_group[g][p] for g in range(n_groups)]) for p in range(phi)]
    path_srs = [sharpe(p) for p in paths]
    corr = np.corrcoef(np.vstack(paths)) if phi > 1 else np.ones((1, 1))
    rho = float(corr[np.triu_indices(phi, 1)].mean()) if phi > 1 else 1.0
    var = float(np.var(path_srs, ddof=1)) if phi > 1 else 0.0
    return {
        "splits": len(splits),
        "paths": phi,
        "path_sharpes": path_srs,
        "mean_sharpe": float(np.mean(path_srs)),
        "std_sharpe": float(np.std(path_srs, ddof=1)) if phi > 1 else 0.0,
        "avg_path_correlation": rho,
        "variance_of_mean": var / phi * (1 + (phi - 1) * rho),
    }


# ── strategy risk (TR-017) ───────────────────────────────────────────────────


def binary_sharpe(p: float, win: float, loss: float, n: float) -> float:
    """Annualised Sharpe of n bets/yr with precision p, payoffs +win / −loss (AFML-15)."""
    mean = p * win - (1 - p) * loss
    sd = (win + loss) * math.sqrt(p * (1 - p))
    return float(mean / sd * math.sqrt(n)) if sd > 0 else math.nan


def implied_precision(target_sr: float, win: float, loss: float, n: float) -> float | None:
    """Precision p solving binary_sharpe = target (the larger root of the quadratic)."""
    span = win + loss  # SR²·span²·p(1−p) = n·(p·span − loss)² → A p² + B p + C = 0
    a = span**2 * (n + target_sr**2)
    b = -(2 * n * span * loss + target_sr**2 * span**2)
    c = n * loss**2
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None
    return float((-b + math.sqrt(disc)) / (2 * a))


def failure_probability(
    bet_returns: np.ndarray, n_per_year: float, target_sr: float, draws: int = 5000, seed: int = 3
) -> dict[str, float | None]:
    """P[p < p_θ]: bootstrap the observed precision; p_θ = implied precision at the target
    Sharpe with the observed average win/loss (standard deviation, not variance — book bug)."""
    wins = bet_returns[bet_returns > 0]
    losses = -bet_returns[bet_returns <= 0]
    if len(wins) == 0 or len(losses) == 0:
        return {"p_threshold": None, "probability_of_failure": None}
    win, loss = float(wins.mean()), float(losses.mean())
    p_theta = implied_precision(target_sr, win, loss, n_per_year)
    rng = np.random.default_rng(seed)
    sample = rng.choice(bet_returns > 0, size=(draws, len(bet_returns)), replace=True).mean(axis=1)
    p_obs = float((bet_returns > 0).mean())
    return {
        "precision": p_obs,
        "p_threshold": p_theta,
        "win": win,
        "loss": loss,
        "probability_of_failure": None if p_theta is None else float((sample < p_theta).mean()),
        "sharpe_from_precision": binary_sharpe(p_obs, win, loss, n_per_year),
    }


# ── synthetic OU trading rules (TR-015) ──────────────────────────────────────


def ou_rule_map(
    forecast: float,
    half_life: float,
    sigma: float,
    take_profits: list[float],
    stop_losses: list[float],
    paths: int = 2000,
    horizon: int = 100,
    seed: int = 9,
) -> np.ndarray:
    """AFML-13: P_t = E + φ(P_{t−1} − E) + σε with φ = 2^(−1/half_life), start at 0, E = forecast;
    exit at profit-take / stop-loss (in σ units) or the horizon; Sharpe of the exit P&L per cell."""
    phi = 2 ** (-1 / half_life)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, sigma, (paths, horizon))
    level = np.zeros((paths, horizon))
    prev = np.zeros(paths)
    for t in range(horizon):
        prev = forecast + phi * (prev - forecast) + shocks[:, t]
        level[:, t] = prev
    out = np.zeros((len(take_profits), len(stop_losses)))
    for i, pt in enumerate(take_profits):
        for j, sl in enumerate(stop_losses):
            hit_pt = level >= pt * sigma
            hit_sl = level <= -sl * sigma
            first_pt = np.where(hit_pt.any(1), hit_pt.argmax(1), horizon)
            first_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), horizon)
            exit_at = np.minimum(np.minimum(first_pt, first_sl), horizon - 1)
            pnl = level[np.arange(paths), exit_at]
            out[i, j] = pnl.mean() / pnl.std(ddof=1) if pnl.std(ddof=1) > 0 else 0.0
    return out


# ── event-driven engine (TR-004) ─────────────────────────────────────────────


def run_event_driven(
    bars: pd.DataFrame,
    target: pd.Series,
    costs: Costs,
    capital: float = 100_000.0,
    participation: float = 0.1,
    borrow_bps: float = 50.0,
    warmup: int = 220,
) -> pd.DataFrame:
    """Orders generated at bar t's close from the target, filled at bar t+1's open; fills capped
    at `participation` × bar volume (the rest carries over: partial fills); fees + slippage on
    traded notional; daily borrow cost on short notional; cash accounting. Returns equity."""
    o, c = bars["open"].to_numpy(dtype=float), bars["close"].to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float) if "volume" in bars else np.full(len(bars), np.inf)
    tgt = target.reindex(bars.index).fillna(0.0).to_numpy()
    cash, shares = capital, 0.0
    rows = []
    for t in range(len(bars)):
        if t >= warmup and t > 0:  # same convention as the vectorised engine at the boundary
            equity_prev = cash + shares * c[t - 1]
            desired = tgt[t - 1] * equity_prev / c[t - 1]
            delta = desired - shares
            cap = participation * vol[t] if np.isfinite(vol[t]) and vol[t] > 0 else abs(delta)
            fill = float(np.sign(delta) * min(abs(delta), cap))
            price = o[t] * (1 + np.sign(fill) * costs.slippage_bps / 10_000)
            cash -= fill * price + abs(fill) * o[t] * costs.fee_bps / 10_000
            shares += fill
        if shares < 0:
            cash -= -shares * c[t] * borrow_bps / 10_000 / 252
        rows.append({"cash": cash, "shares": shares, "equity": cash + shares * c[t]})
    out = pd.DataFrame(rows, index=bars.index)
    out["net"] = out["equity"].pct_change().fillna(0.0)
    return out.iloc[warmup:]


def inject_lookahead(target: pd.Series) -> pd.Series:
    """Test helper for the look-ahead check: a target that already knows tomorrow."""
    return target.shift(-1).fillna(0.0)


def lookahead_check(
    close: pd.Series, target: pd.Series, costs: Costs, warmup: int
) -> dict[str, Any]:
    """Seven-sins look-ahead test: a genuine one-bar-lagged rule keeps most of its Sharpe when
    delayed one more bar; a rule that peeks at the next bar collapses. Flag when the lag-1
    Sharpe is > 2 and loses more than 75 % (and > 1.5 points) at lag 2."""
    sr1 = sharpe(run_vectorised(close, target, costs, warmup, lag=1)["net"].to_numpy())
    sr2 = sharpe(run_vectorised(close, target, costs, warmup, lag=2)["net"].to_numpy())
    flag = bool(math.isfinite(sr1) and sr1 > 2 and (sr1 - sr2) > 1.5 and sr2 < 0.25 * sr1)
    return {"sharpe_lag1": sr1, "sharpe_lag2": sr2, "suspicious": flag}
