"""Carver forecast engine (Quantrick bootcamp 2, after R. Carver's *Systematic Trading* and
*Advanced Futures Trading Strategies*): pure NumPy/pandas.

Each rule gives a forecast scaled so its average absolute value is about 10 and capped at ±20;
rules are combined with weights that sum to 1 and a forecast diversification multiplier (FDM);
the position is F/10 · τ/σ (τ the volatility target, σ the forecast annualised volatility),
long-only by default, capped at 1 (no leverage), with a 10 % no-trade buffer.

Corrections to the course notebook (owner's quant review, QuantRick-Course
`RESULTATS_Bootcamp2_Carver.md`): weights normalised to 1 (the notebook's 0.65 × 1.55 runs the
combined forecast at ≈ 7 instead of 10), FDM from forecast correlations (negatives floored at 0,
capped 2.5) estimated point-in-time on an expanding window instead of fixed, a warm-up before
any position, the position buffer.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

FC_TARGET = 10.0
FC_CAP = 20.0
WARMUP = 260
# (fast span, slow span, forecast scalar) — Carver's published scalars
EWMAC_PAIRS = ((8, 32, 5.95), (16, 64, 4.10), (32, 128, 2.79), (64, 256, 1.91))
BREAKOUT = ((40, 0.70, 10), (80, 0.73, 20), (160, 0.74, 40), (320, 0.74, 80))
ACCEL = ((8, 1.87), (16, 1.90), (32, 1.98), (64, 2.05))  # (lag, scalar) per EWMAC speed
SKEW = ((60, 33.3, 15), (120, 37.2, 30), (240, 39.2, 60))  # (window, scalar, smoothing)
# Fixed within-rule FDMs from the course (sub-variations of one rule are highly correlated)
RULE_FDM = {"ewmac": 1.13, "breakout": 1.17, "accel": 1.55, "skew": 1.18}
RULE_WEIGHTS = {"ewmac": 0.15, "breakout": 0.15, "accel": 0.15, "skew": 0.20}  # normalised below
FDM_MAX = 2.5
FDM_MIN_OBS = 252
FDM_EVERY = 21


def bars_per_year(index: pd.Index) -> float:
    """Annualisation from the calendar itself: ≈ 252 for exchange days, ≈ 365 for crypto."""
    if len(index) < 2:
        return 252.0
    days = (pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).days
    n = (len(index) - 1) / (days / 365.25) if days > 0 else 252.0
    return float(min(max(n, 200.0), 366.0))


def ewm_std(x: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted standard deviation with the small-sample bias correction."""
    a = 2.0 / (span + 1.0)
    m = x.ewm(alpha=a, adjust=False).mean()
    m2 = (x * x).ewm(alpha=a, adjust=False).mean()
    var = (m2 - m * m).clip(lower=0.0)
    return np.sqrt(var / ((2.0 - 2.0 * a) / (2.0 - a)))


def vol_stack(close: pd.Series, periods: float) -> pd.DataFrame:
    """Forecast volatility: 70 % EWMA σ (span 30) + 30 % of its 5-year average (≥ 1 year),
    annualised; `sigma_p` is the same σ in price units per bar (the EWMAC denominator)."""
    ann = math.sqrt(periods)
    ret = close.pct_change().clip(-0.9, 9.0)
    short = ewm_std(ret, 30) * ann
    n = round(periods)
    long = short.rolling(5 * n, min_periods=n).mean()
    vol = pd.Series(np.where(long.isna(), short, 0.7 * short + 0.3 * long), index=close.index)
    return pd.DataFrame({"ret": ret, "vol": vol, "sigma_p": close * vol / ann})


def _mean_capped(subs: list[pd.Series], fdm: float) -> pd.Series:
    return (sum(subs) / len(subs) * fdm).clip(-FC_CAP, FC_CAP)


def ewmac(close: pd.Series, sigma_p: pd.Series) -> pd.Series:
    subs = [_ewmac_base(close, sigma_p, f, s, sc) for f, s, sc in EWMAC_PAIRS]
    return _mean_capped(subs, RULE_FDM["ewmac"])


def _ewmac_base(close: pd.Series, sigma_p: pd.Series, fast: int, slow: int, sc: float) -> pd.Series:
    diff = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return (diff / sigma_p * sc).clip(-FC_CAP, FC_CAP)


def breakout(close: pd.Series) -> pd.Series:
    """Position of the close inside its n-day range, centred and smoothed."""
    subs = []
    for n, sc, smooth in BREAKOUT:
        lo, hi = close.rolling(n).min(), close.rolling(n).max()
        rng = hi - lo
        where = ((close - lo) / rng).where(rng > 0, 0.5)
        raw = ((where - 0.5) * 40.0).ewm(span=smooth, adjust=False).mean()
        subs.append((raw * sc).clip(-FC_CAP, FC_CAP))
    return _mean_capped(subs, RULE_FDM["breakout"])


def acceleration(close: pd.Series, sigma_p: pd.Series) -> pd.Series:
    """Change of each EWMAC forecast over its fast span: the trend is strengthening."""
    subs = []
    for (f, s, sc), (lag, scalar) in zip(EWMAC_PAIRS, ACCEL, strict=True):
        base = _ewmac_base(close, sigma_p, f, s, sc)
        subs.append(((base - base.shift(lag)) * scalar).clip(-FC_CAP, FC_CAP))
    return _mean_capped(subs, RULE_FDM["accel"])


def skew(ret: pd.Series) -> pd.Series:
    """Negative of the rolling skewness: prefer assets whose returns are skewed to the right."""
    subs = []
    for window, sc, smooth in SKEW:
        g = ret.rolling(window, min_periods=window // 2).skew()
        subs.append((-g).ewm(span=smooth, adjust=False).mean().mul(sc).clip(-FC_CAP, FC_CAP))
    return _mean_capped(subs, RULE_FDM["skew"])


def forecasts(close: pd.Series, periods: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The four rule forecasts (NaN during the warm-up) and the volatility frame."""
    p = periods or bars_per_year(close.index)
    v = vol_stack(close, p)
    f = pd.DataFrame(
        {
            "ewmac": ewmac(close, v["sigma_p"]),
            "breakout": breakout(close),
            "accel": acceleration(close, v["sigma_p"]),
            "skew": skew(v["ret"]),
        }
    )
    f.iloc[:WARMUP] = np.nan
    return f, v


def weights(rules: Sequence[str] | None = None) -> np.ndarray:
    """Rule weights normalised to 1 (over `rules`, in their order; all rules by default)."""
    names = list(RULE_WEIGHTS) if rules is None else list(rules)
    w = np.array([RULE_WEIGHTS[k] for k in names], dtype=float)
    out: np.ndarray = w / w.sum()
    return out


def fdm_from_corr(w: np.ndarray, corr: np.ndarray) -> float:
    """1 / √(w'Cw) with negative correlations floored at 0, capped at FDM_MAX."""
    c = np.clip(np.nan_to_num(corr, nan=0.0), 0.0, None)
    np.fill_diagonal(c, 1.0)
    var = float(w @ c @ w)
    return float(min(FDM_MAX, 1.0 / math.sqrt(var))) if var > 0 else 1.0


def fdm_path(f: pd.DataFrame) -> pd.Series:
    """Point-in-time FDM: re-estimated every month on all forecasts up to that bar (expanding),
    1 until a year of forecasts exists — never uses a later forecast."""
    w = weights(list(f.columns))
    if len(w) == 1:  # one rule: nothing to diversify
        return pd.Series(1.0, index=f.index)
    values = f.to_numpy()
    out = np.ones(len(f))
    current = 1.0
    valid = ~np.isnan(values).any(axis=1)
    seen = np.cumsum(valid)
    for i in range(len(f)):
        if seen[i] >= FDM_MIN_OBS and (i % FDM_EVERY == 0):
            past = values[: i + 1][valid[: i + 1]]
            current = fdm_from_corr(w, np.corrcoef(past, rowvar=False))
        out[i] = current
    return pd.Series(out, index=f.index)


def combined(f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Weighted forecast × point-in-time FDM, capped ±20; returns (forecast, FDM path)."""
    fdm = fdm_path(f)
    raw = f.mul(weights(list(f.columns)), axis=1).sum(axis=1, min_count=f.shape[1])
    return (raw * fdm).clip(-FC_CAP, FC_CAP), fdm


def apply_buffer(target: pd.Series, base: pd.Series, frac: float) -> pd.Series:
    """No-trade band: keep the held position while it stays within target ± frac·base."""
    if frac <= 0:
        return target
    t, b = target.to_numpy(), (frac * base).fillna(0.0).to_numpy()
    out = np.zeros(len(t))
    cur = 0.0
    for i in range(len(t)):
        lo, hi = t[i] - b[i], t[i] + b[i]
        cur = lo if cur < lo else hi if cur > hi else cur
        out[i] = cur
    return pd.Series(out, index=target.index)


def position(
    close: pd.Series,
    vol_target: float = 0.20,
    buffer: float = 0.10,
    long_only: bool = True,
    rules: Sequence[str] | None = None,
) -> pd.Series:
    """Target weight on each close (the backtest engine shifts it one bar). `rules` keeps a
    subset of the four rules, weights renormalised (ablation, TR-024); all rules by default."""
    f, v = forecasts(close)
    if rules is not None:
        f = f[list(rules)]
    fc, _ = combined(f)
    scale = (vol_target / v["vol"]).replace([np.inf, -np.inf], np.nan)
    w = (fc / FC_TARGET * scale).fillna(0.0).clip(0.0 if long_only else -1.0, 1.0)
    base = scale.clip(upper=1.0).where(f.notna().all(axis=1), 0.0)
    return apply_buffer(w, base, buffer).clip(0.0 if long_only else -1.0, 1.0)


def vol_targeted_hold(close: pd.Series, vol_target: float = 0.20) -> pd.Series:
    """The honest benchmark: always long, sized by the same volatility (no trend), capped at 1."""
    f, v = forecasts(close)
    w = (vol_target / v["vol"]).replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
    return w.where(f.notna().all(axis=1), 0.0).fillna(0.0)


def prop_challenge(
    daily: np.ndarray,
    vol_targets: tuple[float, ...],
    periods: float = 252.0,
    n_paths: int = 5000,
    block: int = 20,
    horizon: int = 250,
    targets: tuple[float, float] = (0.10, 0.05),
    daily_loss: float = 0.05,
    max_loss: float = 0.10,
    min_days: int = 4,
    seed: int = 1,
) -> list[dict[str, float | None]]:
    """Two-phase prop-firm challenge by block bootstrap of daily returns (blocks keep the loss
    clustering that resampling single days or trades breaks). The returns are rescaled to each
    volatility target; a phase passes at +target with ≥ min_days traded, fails on a daily loss
    ≥ daily_loss or equity ≤ 1 − max_loss of the initial capital, and expires after horizon days.
    Phase 2 restarts from the initial capital. Close-to-close: real challenges check intraday
    losses, so these probabilities are a little optimistic."""
    r = np.asarray(daily, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 * block:
        raise ValueError("Not enough daily returns for the bootstrap")
    base_vol = float(r.std(ddof=1) * math.sqrt(periods))
    rng = np.random.default_rng(seed)
    n_blocks = 2 * horizon // block + 2
    starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
    paths = r[(starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)][:, : 2 * horizon]
    rows: list[dict[str, float | None]] = []
    for tv in vol_targets:
        x = paths * (tv / base_vol if base_vol > 0 else 0.0)
        phase = np.zeros(n_paths, dtype=int)
        eq = np.ones(n_paths)
        days_in = np.zeros(n_paths, dtype=int)
        alive = np.ones(n_paths, dtype=bool)
        funded = np.zeros(n_paths, dtype=bool)
        failed_p1 = np.zeros(n_paths, dtype=bool)
        days = np.full(n_paths, np.nan)
        for d in range(2 * horizon):
            act = alive & ~funded
            if not act.any():
                break
            pnl = eq * x[:, d]
            eq = np.where(act, eq + pnl, eq)
            days_in = np.where(act, days_in + 1, days_in)
            fail = act & ((pnl <= -daily_loss) | (eq <= 1.0 - max_loss))
            failed_p1 |= fail & (phase == 0)
            alive &= ~fail
            goal = np.where(phase == 0, 1.0 + targets[0], 1.0 + targets[1])
            hit = act & ~fail & (eq >= goal) & (days_in >= min_days)
            first, second = hit & (phase == 0), hit & (phase == 1)
            funded |= second
            days[second] = d + 1
            phase[first], eq[first], days_in[first] = 1, 1.0, 0
            alive &= ~(act & ~fail & ~hit & (days_in >= horizon))
        rows.append(
            {
                "vol_target": tv,
                "p_funded": float(funded.mean()),
                "p_fail_phase1": float(failed_p1.mean()),
                "median_days_if_funded": float(np.nanmedian(days)) if funded.any() else None,
            }
        )
    return rows


# ── multi-asset book with IDM (TR-026) ───────────────────────────────────────

IDM_MAX = 2.5


def idm_path(sub: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Point-in-time instrument diversification multiplier, 1 / √(w'Hw) on the correlations of
    the instruments' sub-system returns (negatives floored at 0, cap 2.5): re-estimated every
    month on all returns up to that bar, 1 until a year of returns exists."""
    values = sub.to_numpy()
    out = np.ones(len(sub))
    current = 1.0
    live = (values != 0).any(axis=1)
    seen = np.cumsum(live)
    for i in range(len(sub)):
        if seen[i] >= FDM_MIN_OBS and (i % FDM_EVERY == 0):
            past = values[: i + 1][live[: i + 1]]
            current = min(IDM_MAX, fdm_from_corr(w, np.corrcoef(past, rowvar=False)))
        out[i] = current
    return pd.Series(out, index=sub.index)


def book_positions(
    returns: pd.DataFrame,
    weights: np.ndarray,
    per_asset: Callable[[pd.Series], pd.Series],
    max_leverage: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Each instrument sized on its own (`per_asset` on its total-return index), then
    position_i = IDM · w_i · p_i, decided at the close; the gross exposure is scaled down to
    `max_leverage` when above it. Returns (positions, IDM path)."""
    own = pd.DataFrame(
        {c: per_asset((1 + returns[c].fillna(0.0)).cumprod()) for c in returns.columns}
    )
    sub = own.shift(1).fillna(0.0) * returns.fillna(0.0)
    idm = idm_path(sub, weights)
    raw = own.mul(weights, axis=1).mul(idm, axis=0)
    gross = raw.abs().sum(axis=1)
    scale = (max_leverage / gross).where(gross > max_leverage, 1.0)
    return raw.mul(scale, axis=0), idm


def run_book(
    returns: pd.DataFrame, positions: pd.DataFrame, cost_per_unit: float, warmup: int
) -> pd.DataFrame:
    """net_t = Σ_i p_{i,t−1}·r_{i,t} − cost · Σ_i |p_{i,t−1} − p_{i,t−2}| (one-bar shift, costs
    on traded notional, the warm-up excluded): the vectorised engine for a book."""
    held = positions.shift(1).fillna(0.0)
    held.iloc[:warmup] = 0.0
    traded = held.diff().abs().fillna(held.abs()).sum(axis=1)
    gross = (held * returns.fillna(0.0)).sum(axis=1)
    frame = pd.DataFrame(
        {
            "position": held.abs().sum(axis=1),
            "gross": gross,
            "cost": traded * cost_per_unit,
            "net": gross - traded * cost_per_unit,
            "asset": returns.fillna(0.0).mean(axis=1),
        }
    )
    return frame.iloc[warmup:]
