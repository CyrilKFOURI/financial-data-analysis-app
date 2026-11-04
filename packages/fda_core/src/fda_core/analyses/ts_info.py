"""Information & microstructure from daily bars (TS-013 entropy, TS-016 microstructure;
AFML 18–19). Intraday-only features (alternative bars TS-015, VPIN) need the intraday
dataset and arrive with it (J9)."""

import math
from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from fda_core.analyses.price import PRICE_SUBJECTS, _annualisation, _field, _prices, _windowed
from fda_core.analyses.ts_common import SeriesParams, clean, iso, require
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

# ── encodings & estimators (AFML 18.2–18.5) ──────────────────────────────────

Encoding = Literal["binary", "quantile", "sigma"]


def encode(returns: np.ndarray, encoding: str, bins: int = 10) -> np.ndarray:
    """Binary: sign; quantile: equal-count bins; sigma: bins of width σ (clipped to ±5)."""
    if encoding == "binary":
        return (returns > 0).astype(int)
    if encoding == "quantile":
        edges = np.quantile(returns, np.linspace(0, 1, bins + 1)[1:-1])
        return np.searchsorted(edges, returns)
    sigma = returns.std(ddof=1)
    return np.clip(np.floor(returns / sigma), -5, 5).astype(int) + 5


def plug_in_entropy(message: np.ndarray, word: int = 1) -> float:
    """Maximum-likelihood entropy rate in bits per symbol over words of length `word`."""
    words = [tuple(message[i : i + word]) for i in range(len(message) - word + 1)]
    counts = np.array(list(Counter(words).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() / word)


def lempel_ziv_entropy(message: np.ndarray) -> float:
    """LZ76 complexity c(n) normalised as c·log2(n)/n (→ entropy rate in bits for binary data)."""
    s = "".join(chr(65 + int(v)) for v in message)
    n = len(s)
    i, c = 0, 0
    while i < n:
        length = 1
        while i + length <= n and s[i : i + length] in s[: i + length - 1]:
            length += 1
        c += 1
        i += length
    return c * math.log2(n) / n


def kontoyiannis_entropy(message: np.ndarray, window: int | None = None) -> float:
    """Match-length estimator: H ≈ [mean_i L_i / log2(n_i)]⁻¹ with L_i the shortest substring
    starting at i not found in the preceding window (AFML 18.4)."""
    s = "".join(chr(65 + int(v)) for v in message)
    n = len(s)
    start = n // 2 if window is None else window
    lengths = []
    for i in range(start, n):
        past = s[max(0, i - (window or i)) : i]
        length = 1
        while i + length <= n and s[i : i + length] in past:
            length += 1
        n_i = len(past)
        if n_i > 1:
            lengths.append(length / math.log2(n_i))
    return float(1 / np.mean(lengths)) if lengths else float("nan")


class EntropyParams(SeriesParams):
    encoding: Encoding = "binary"
    word: int = Field(2, ge=1, le=6)
    window: int = Field(252, ge=100, le=1_260, description="Rolling window (sessions)")


@analysis(
    id="TS-013",
    version="1.0.0",
    title="Entropy & information",
    description=(
        "Shannon entropy rate of encoded returns by plug-in (word length w), Lempel–Ziv and "
        "Kontoyiannis estimators, as a share of the maximum log2(alphabet); rolling plug-in "
        "entropy as an efficiency proxy (high = less predictable). Binary/quantile/sigma "
        "encodings (AFML-18)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=EntropyParams,
    result_kind="table",
    requires={"prices"},
    min_obs=100,
    budget_ms=10_000,
)
def entropy(inputs: AnalysisInputs, params: EntropyParams) -> AnalysisOutput:
    _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    r = require(returns, 100, "Entropy estimators")
    message = encode(r, params.encoding)
    alphabet = len(np.unique(message))
    max_bits = math.log2(alphabet) if alphabet > 1 else 1.0
    estimates = {
        "plug_in": plug_in_entropy(message, params.word),
        "lempel_ziv": lempel_ziv_entropy(message) if params.encoding == "binary" else None,
        "kontoyiannis": kontoyiannis_entropy(message[-1_000:]),
    }
    rolling_dates, rolling_values = [], []
    for end in range(params.window, len(message) + 1, 21):
        rolling_dates.append(iso(returns.index[[end - 1]])[0])
        rolling_values.append(
            plug_in_entropy(message[end - params.window : end], params.word) / max_bits
        )
    return AnalysisOutput(
        "table",
        {
            "encoding": params.encoding,
            "alphabet": alphabet,
            "max_bits": max_bits,
            "estimates": estimates,
            "normalised": {
                k: (v / max_bits if v is not None and np.isfinite(v) else None)
                for k, v in estimates.items()
            },
            "rolling": {"dates": rolling_dates, "values": rolling_values},
        },
        ("Kontoyiannis uses the last 1 000 symbols (quadratic cost).",),
    )


# ── TS-016 microstructure (daily) ────────────────────────────────────────────


class MicrostructureParams(SeriesParams):
    window: int = Field(63, ge=21, le=252)


def roll_spread(close: pd.Series, window: int) -> pd.Series:
    """Roll (1984): 2√(−cov(Δp_t, Δp_{t−1})), NaN when the covariance is positive."""
    dp = close.diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return 2 * np.sqrt((-cov).clip(lower=0)).where(cov < 0)


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin–Schultz (2012) two-day high–low spread estimator, negatives set to 0."""
    beta = np.log(high / low) ** 2
    beta = beta + beta.shift(1)
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def rolling_slope(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """OLS slope of y on x through the origin over trailing windows."""
    return (x * y).rolling(window).sum() / (x * x).rolling(window).sum()


@analysis(
    id="TS-016",
    version="1.0.0",
    title="Microstructure features",
    description=(
        "From daily bars: tick-rule signs, Roll spread, Corwin–Schultz spread, Amihud λ = "
        "mean |r| / dollar volume, Kyle λ (Δp on signed volume) and Hasbrouck λ (r on signed "
        "√(dollar volume)) over rolling windows, with current percentile ranks. VPIN and PIN "
        "need intraday trades (J9)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=MicrostructureParams,
    result_kind="series",
    requires={"prices"},
    min_obs=63,
)
def microstructure(inputs: AnalysisInputs, params: MicrostructureParams) -> AnalysisOutput:
    if not inputs.conventions.get("has_volume", True):
        raise InsufficientDataError("Microstructure λ need meaningful volume")
    full = _prices(inputs)
    close, volume = full["close"].astype(float), full["volume"].astype(float)
    require(close, params.window + 2, "Microstructure features")
    dp = close.diff()
    sign = np.sign(dp).replace(0, np.nan).ffill().fillna(0)  # tick rule
    returns = close.pct_change()
    dollar = close * volume
    amihud = (returns.abs() / dollar.replace(0, np.nan)).rolling(params.window).mean() * 1e6
    kyle = rolling_slope(dp, sign * volume, params.window)
    hasbrouck = rolling_slope(returns, sign * np.sqrt(dollar), params.window)
    features = {
        "roll_spread": roll_spread(close, params.window) / close,
        "corwin_schultz": corwin_schultz(full["high"].astype(float), full["low"].astype(float))
        .rolling(params.window)
        .mean(),
        "amihud_per_million": amihud,
        "kyle_lambda": kyle,
        "hasbrouck_lambda": hasbrouck,
    }
    visible = _windowed(inputs, full).index
    out: dict[str, Any] = {"dates": iso(visible), "lines": {}, "current": {}}
    for name, series in features.items():
        out["lines"][name] = clean(series.loc[visible])
        history = series.dropna()
        last = series.loc[visible].dropna()
        out["current"][name] = {
            "value": float(last.iloc[-1]) if len(last) else None,
            "percentile": float((history <= last.iloc[-1]).mean())
            if len(last) and len(history)
            else None,
        }
    return AnalysisOutput("series", out, ("Spreads are relative to price; λ in price units.",))
