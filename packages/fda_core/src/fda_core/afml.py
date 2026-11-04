"""AFML toolkit (M27, TR-021): pure NumPy/SciPy implementations of López de Prado ch. 2–10.

Includes the owner's notebook corrections: average uniqueness attributes returns from the bar
AFTER the event (the event day is excluded), purging is applied around each test fold, and
probabilities are calibrated before bet sizing. Models available without new dependencies: an
L2 logistic regression and a k-nearest-neighbour classifier (tree ensembles need scikit-learn).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats

Array = NDArray[np.float64]


# ── events & labels (AFML-2/3) ───────────────────────────────────────────────


def cusum_events(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns: an event when the positive or negative
    cumulative sum exceeds the threshold (scalar or per-bar series), then reset."""
    r = np.log(close).diff().dropna()
    thr = (
        threshold.reindex(r.index).ffill().to_numpy()
        if isinstance(threshold, pd.Series)
        else np.full(len(r), threshold)
    )
    events, s_pos, s_neg = [], 0.0, 0.0
    for t, (x, h) in enumerate(zip(r.to_numpy(), thr, strict=True)):
        if not math.isfinite(h):
            continue
        s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
        if s_pos > h:
            s_pos = 0.0
            events.append(r.index[t])
        elif s_neg < -h:
            s_neg = 0.0
            events.append(r.index[t])
    return pd.DatetimeIndex(events)


def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    return np.log(close).diff().ewm(span=span).std()


def triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    vol: pd.Series,
    pt: float = 1.0,
    sl: float = 1.0,
    horizon: int = 10,
) -> pd.DataFrame:
    """For each event t0: upper = +pt·σ, lower = −sl·σ on the log return from t0, vertical =
    t0 + horizon bars. Label = sign of the first touched barrier (vertical: sign of the return).
    Columns: t1 (exit), ret, label (−1/0/1), barrier ('pt' | 'sl' | 'vertical')."""
    logp = np.log(close)
    idx = close.index
    rows = []
    for t0 in events:
        if t0 not in vol.index or not math.isfinite(vol.loc[t0]) or vol.loc[t0] <= 0:
            continue
        i0 = idx.get_loc(t0)
        end = min(i0 + horizon, len(idx) - 1)
        if end <= i0:
            continue
        path = logp.iloc[i0 + 1 : end + 1] - logp.iloc[i0]
        up, down = pt * vol.loc[t0], -sl * vol.loc[t0]
        hit_up = path.index[path >= up] if pt > 0 else pd.DatetimeIndex([])
        hit_dn = path.index[path <= down] if sl > 0 else pd.DatetimeIndex([])
        first_up = hit_up[0] if len(hit_up) else None
        first_dn = hit_dn[0] if len(hit_dn) else None
        if first_up is not None and (first_dn is None or first_up <= first_dn):
            t1, barrier = first_up, "pt"
        elif first_dn is not None:
            t1, barrier = first_dn, "sl"
        else:
            t1, barrier = idx[end], "vertical"
        ret = float(np.exp(logp.loc[t1] - logp.loc[t0]) - 1)
        rows.append(
            {
                "t0": t0,
                "t1": t1,
                "ret": ret,
                "label": int(np.sign(ret))
                if barrier == "vertical"
                else (1 if barrier == "pt" else -1),
                "barrier": barrier,
            }
        )
    return (
        pd.DataFrame(rows).set_index("t0")
        if rows
        else pd.DataFrame(columns=["t1", "ret", "label", "barrier"])
    )


# ── sample weights (AFML-4) ──────────────────────────────────────────────────


def concurrency(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Number of labels alive on each bar; a label spans (t0, t1] — the event day t0 is not
    part of its return attribution (notebook fix)."""
    count = pd.Series(0, index=index, dtype=float)
    for t0, end in t1.items():
        count.loc[(count.index > t0) & (count.index <= end)] += 1
    return count


def average_uniqueness(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """ū_i = mean over (t0, t1] of 1 / c_t."""
    c = concurrency(index, t1)
    out = {}
    for t0, end in t1.items():
        span = c.loc[(c.index > t0) & (c.index <= end)]
        out[t0] = float((1 / span).mean()) if len(span) else np.nan
    return pd.Series(out)


def indicator_matrix(index: pd.DatetimeIndex, t1: pd.Series) -> Array:
    m = np.zeros((len(index), len(t1)))
    pos = {d: i for i, d in enumerate(index)}
    for j, (t0, end) in enumerate(t1.items()):
        start = index.searchsorted(t0, side="right")
        stop = pos.get(end, index.searchsorted(end))
        m[start : stop + 1, j] = 1.0
    return m


def sequential_bootstrap(ind: Array, size: int | None = None, seed: int = 0) -> list[int]:
    """Draw labels with probability proportional to their average uniqueness given those
    already drawn (AFML-4.5.3)."""
    rng = np.random.default_rng(seed)
    n = ind.shape[1]
    size = size or n
    chosen: list[int] = []
    load = np.zeros(ind.shape[0])
    for _ in range(size):
        active = ind > 0
        u = np.array(
            [
                float((ind[active[:, j], j] / (load[active[:, j]] + 1)).mean())
                if active[:, j].any()
                else 0.0
                for j in range(n)
            ]
        )
        prob = u / u.sum()
        j = int(rng.choice(n, p=prob))
        chosen.append(j)
        load += ind[:, j]
    return chosen


# ── cross-validation (AFML-7) ────────────────────────────────────────────────


def purged_kfold(
    t0: pd.DatetimeIndex,
    t1: pd.Series,
    n_splits: int = 5,
    embargo: float = 0.01,
    shuffle: bool = False,
    seed: int = 0,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Contiguous test folds; training labels whose (t0, t1] overlaps the test span are purged,
    and those starting within the embargo after it removed. `shuffle=True` gives the naive
    shuffled K-fold (for the leakage demonstration)."""
    n = len(t0)
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n), n_splits)
        return [(np.setdiff1d(order, f), np.sort(f)) for f in folds]
    folds = np.array_split(order, n_splits)
    gap = math.ceil(n * embargo)
    t1_arr = t1.reindex(t0).to_numpy()
    out = []
    for f in folds:
        test_start, test_end = t0[f[0]], t1_arr[f].max()
        overlap = (t0 <= test_end) & (pd.DatetimeIndex(t1_arr) >= test_start)
        embargoed = np.zeros(n, dtype=bool)
        embargoed[f[-1] + 1 : f[-1] + 1 + gap] = True
        train = order[~overlap & ~embargoed & ~np.isin(order, f)]
        out.append((train, f))
    return out


# ── models (numpy) ───────────────────────────────────────────────────────────


@dataclass
class Logistic:
    """L2-penalised logistic regression (standardised inputs), fitted by L-BFGS."""

    c: float = 1.0
    coef: Array | None = None
    mean: Array | None = None
    scale: Array | None = None

    def fit(self, x: Array, y: Array, w: Array | None = None) -> "Logistic":
        self.mean, self.scale = x.mean(0), x.std(0) + 1e-12
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        w = np.ones(len(y)) if w is None else w / w.mean()

        def loss(beta: Array) -> tuple[float, Array]:
            p = 1 / (1 + np.exp(-np.clip(z @ beta, -30, 30)))
            ll = -np.sum(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            reg = 0.5 / self.c * np.sum(beta[1:] ** 2)
            grad = z.T @ (w * (p - y)) + np.r_[0.0, beta[1:] / self.c]
            return float(ll + reg), grad

        self.coef = optimize.minimize(loss, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def proba(self, x: Array) -> Array:
        assert self.coef is not None
        assert self.mean is not None
        assert self.scale is not None
        z = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        return np.asarray(1 / (1 + np.exp(-np.clip(z @ self.coef, -30, 30))))


@dataclass
class NearestNeighbours:
    """k-NN on standardised features: the probability of class 1 among the k nearest."""

    k: int = 1
    x: Array | None = None
    y: Array | None = None
    mean: Array | None = None
    scale: Array | None = None

    def fit(self, x: Array, y: Array, w: Array | None = None) -> "NearestNeighbours":
        self.mean, self.scale = x.mean(0), x.std(0) + 1e-12
        self.x, self.y = (x - self.mean) / self.scale, y
        return self

    def proba(self, x: Array) -> Array:
        assert self.x is not None
        assert self.y is not None
        assert self.mean is not None
        assert self.scale is not None
        z = (x - self.mean) / self.scale
        d = ((z[:, None, :] - self.x[None, :, :]) ** 2).sum(-1)
        nearest = np.argsort(d, axis=1)[:, : self.k]
        return np.asarray(self.y[nearest].mean(1), dtype=float)


def cv_scores(
    model_factory: type[Logistic] | type[NearestNeighbours],
    x: Array,
    y: Array,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    w: Array | None = None,
) -> dict[str, float]:
    """Out-of-fold accuracy, log-loss, precision, F1 and base rates."""
    proba = np.full(len(y), np.nan)
    for train, test in splits:
        model = model_factory().fit(x[train], y[train], None if w is None else w[train])
        proba[test] = model.proba(x[test])
    ok = np.isfinite(proba)
    p, t = np.clip(proba[ok], 1e-6, 1 - 1e-6), y[ok]
    pred = p >= 0.5
    tp = float((pred & (t == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / (t == 1).sum() if (t == 1).sum() else 0.0
    base = float(t.mean())
    return {
        "accuracy": float((pred == t).mean()),
        "log_loss": float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))),
        "naive_log_loss": float(-(base * math.log(base) + (1 - base) * math.log(1 - base)))
        if 0 < base < 1
        else math.nan,
        "precision": precision,
        "base_rate": base,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "f1_trivial": 2 * base / (1 + base),
        "n": int(ok.sum()),
    }


def mda(
    x: Array,
    y: Array,
    splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    names: list[str],
    seed: int = 0,
) -> list[dict[str, float | str]]:
    """Mean decrease accuracy (OOS, purged folds): log-loss increase when a feature is
    permuted in the test fold (substitution effects can hide correlated features)."""
    rng = np.random.default_rng(seed)
    base_losses: list[float] = []
    drops: dict[str, list[float]] = {n: [] for n in names}
    for train, test in splits:
        model = Logistic().fit(x[train], y[train])
        p = np.clip(model.proba(x[test]), 1e-6, 1 - 1e-6)
        base = -np.mean(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p))
        base_losses.append(base)
        for j, n in enumerate(names):
            xp = x[test].copy()
            xp[:, j] = rng.permutation(xp[:, j])
            pp = np.clip(model.proba(xp), 1e-6, 1 - 1e-6)
            drops[n].append(-np.mean(y[test] * np.log(pp) + (1 - y[test]) * np.log(1 - pp)) - base)
    return [
        {"feature": n, "mda": float(np.mean(v)), "se": float(np.std(v, ddof=1) / math.sqrt(len(v)))}
        for n, v in drops.items()
    ]


def sfi(
    x: Array, y: Array, splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]], names: list[str]
) -> list[dict[str, float | str]]:
    """Single feature importance: OOS log-loss of a model on each feature alone (no
    substitution effects)."""
    return [
        {"feature": n, "log_loss": cv_scores(Logistic, x[:, [j]], y, splits)["log_loss"]}
        for j, n in enumerate(names)
    ]


# ── calibration & bet sizing (AFML-10) ───────────────────────────────────────


def platt(scores: Array, y: Array) -> tuple[float, float]:
    model = Logistic(c=1e6).fit(scores.reshape(-1, 1), y)
    assert model.coef is not None
    assert model.mean is not None
    assert model.scale is not None
    return float(model.coef[1] / model.scale[0]), float(
        model.coef[0] - model.coef[1] * model.mean[0] / model.scale[0]
    )


def bet_size(p: Array, step: float = 0.0) -> Array:
    """m = 2Φ(z) − 1 with z = (p − ½)/√(p(1 − p)) (two classes), optionally discretised."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    m = 2 * stats.norm.cdf((p - 0.5) / np.sqrt(p * (1 - p))) - 1
    if step > 0:
        m = np.round(m / step) * step
    return np.asarray(np.clip(m, -1, 1))


def average_active_bets(t1: pd.Series, sizes: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Position on each bar = mean of the sizes of the bets alive on it (AFML-10.3)."""
    out = pd.Series(0.0, index=index)
    count = pd.Series(0.0, index=index)
    for t0, end in t1.items():
        alive = (index > t0) & (index <= end)
        out[alive] += sizes.loc[t0]
        count[alive] += 1
    return (out / count.replace(0, np.nan)).fillna(0.0)
