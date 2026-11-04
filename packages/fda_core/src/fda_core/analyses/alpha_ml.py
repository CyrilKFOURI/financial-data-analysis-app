"""Alpha research, financial ML & forecasting (ANALYTICS_CATALOG TR-020…TR-023; module M27).

Models available without new dependencies: L2 logistic regression and k-NN (fda_core.afml),
ARIMA / exponential smoothing (statsmodels). Tree ensembles (scikit-learn) and foundation
forecasters (Chronos-2, TimesFM) need the owner's approval to add their dependencies; the
payloads say so instead of pretending. Every output is a labelled model output.
"""

import math
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from fda_core import afml
from fda_core import backtest as bt
from fda_core.analyses.price import PRICE_SUBJECTS, _field, _prices
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

LABEL = "Model output — research evidence, not a recommendation"
PENDING = "Random forest / gradient boosting need scikit-learn (dependency awaiting approval)"


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── TR-020 alpha factor lab ──────────────────────────────────────────────────

Factor = Literal["momentum_12_1", "reversal_1m", "low_volatility", "trend_200d"]


def factor_panel(prices: pd.DataFrame, factor: Factor) -> pd.DataFrame:
    """Factor values per date × asset from prices only (point in time by construction)."""
    if factor == "momentum_12_1":
        return prices.shift(21) / prices.shift(252) - 1
    if factor == "reversal_1m":
        return -(prices / prices.shift(21) - 1)
    if factor == "low_volatility":
        return -np.log(prices).diff().rolling(252, min_periods=200).std()
    return prices / prices.rolling(200, min_periods=200).mean() - 1


class AlphaParams(BaseModel):
    factor: Factor = "momentum_12_1"
    horizon: int = Field(21, ge=1, le=126)
    quantiles: int = Field(5, ge=2, le=10)
    cost_bps: float = Field(10.0, ge=0, le=100)


@analysis(
    id="TR-020",
    version="1.0.0",
    title="Alpha factor lab",
    description=(
        "Factor computed from member prices (12-1 momentum, 1-month reversal, low volatility, "
        "distance to the 200-day average), sampled monthly: Spearman rank IC with forward returns "
        "at the horizon (mean, IR, t), IC decay at 1/5/21/63 days, quantile mean forward returns, "
        "rank autocorrelation (turnover proxy), and the top-minus-bottom quantile portfolio with "
        "costs on rank changes."
    ),
    applies_to={"universe"},
    params=AlphaParams,
    result_kind="table",
    requires={"member_prices"},
    budget_ms=60_000,
)
def alpha_lab(inputs: AnalysisInputs, params: AlphaParams) -> AnalysisOutput:
    members = inputs.data.get("member_prices")
    if not isinstance(members, dict) or len(members) < 10:
        raise InsufficientDataError("The alpha lab needs stored prices for ≥ 10 members")
    prices = pd.concat(members, axis=1).sort_index().ffill(limit=5)
    values = factor_panel(prices, params.factor)
    month_ends = prices.resample("ME").last().index
    dates = [
        prices.index[prices.index.searchsorted(d, side="right") - 1]
        for d in month_ends
        if d >= prices.index[0]
    ]
    dates = [d for d in dates if values.loc[d].notna().sum() >= 10]
    ics: dict[int, list[float]] = {h: [] for h in (1, 5, 21, 63)}
    quant_rets: list[np.ndarray] = []
    ls, prev_rank = [], None
    autocorr = []
    for d in dates:
        i = prices.index.get_loc(d)
        f = values.loc[d]
        for h in ics:
            if i + h < len(prices):
                fwd = prices.iloc[i + h] / prices.iloc[i] - 1
                both = pd.concat([f, fwd], axis=1).dropna()
                if len(both) >= 10:
                    ics[h].append(float(stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]))
        if i + params.horizon >= len(prices):
            continue
        fwd = prices.iloc[i + params.horizon] / prices.iloc[i] - 1
        both = pd.concat([f.rename("f"), fwd.rename("r")], axis=1).dropna()
        if len(both) < params.quantiles * 2:
            continue
        both["q"] = pd.qcut(both["f"].rank(method="first"), params.quantiles, labels=False)
        q_mean = both.groupby("q")["r"].mean().to_numpy()
        quant_rets.append(q_mean)
        rank = both["f"].rank(pct=True)
        cost = 0.0
        if prev_rank is not None:
            common = rank.index.intersection(prev_rank.index)
            if len(common) > 5:
                autocorr.append(float(stats.spearmanr(rank[common], prev_rank[common])[0]))
                top_now = set(both.index[both["q"] == params.quantiles - 1])
                cost = params.cost_bps / 10_000 * 2 * (1 - (autocorr[-1] if autocorr else 0))
                del top_now
        ls.append(q_mean[-1] - q_mean[0] - cost)
        prev_rank = rank
    main = ics[params.horizon] if params.horizon in ics else ics[21]
    if len(main) < 12:
        raise InsufficientDataError(
            f"Needs ≥ 12 monthly cross-sections with ≥ 10 members, got {len(main)}"
        )
    ic = np.array(main)
    q = np.array(quant_rets)
    ls_arr = np.array(ls)
    periods = 12 * 21 / params.horizon if params.horizon <= 21 else 12
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "factor": params.factor,
            "horizon": params.horizon,
            "members": len(members),
            "months": len(ic),
            "ic_mean": float(ic.mean()),
            "ic_ir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else None,
            "ic_t": float(ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)))
            if ic.std(ddof=1) > 0
            else None,
            "ic_positive_share": float((ic > 0).mean()),
            "decay": [
                {"horizon": h, "ic_mean": _f(np.mean(v)) if v else None, "n": len(v)}
                for h, v in ics.items()
            ],
            "quantiles": [
                {"quantile": i + 1, "mean_forward_return": float(v)}
                for i, v in enumerate(q.mean(0))
            ],
            "rank_autocorrelation": _f(np.mean(autocorr)) if autocorr else None,
            "long_short_mean": float(ls_arr.mean()),
            "long_short_sharpe": float(ls_arr.mean() / ls_arr.std(ddof=1) * math.sqrt(periods))
            if ls_arr.std(ddof=1) > 0
            else None,
            "survivorship": "Universe = today's members: backward results are survivorship-biased",
        },
    )


# ── TR-021 labels & weights / TR-022 ML pipeline ─────────────────────────────


class LabelParams(BaseModel):
    pt: float = Field(1.0, ge=0, le=10, description="Profit-take, multiple of daily σ")
    sl: float = Field(1.0, ge=0, le=10, description="Stop-loss, multiple of daily σ")
    horizon: int = Field(10, ge=1, le=126, description="Vertical barrier (bars)")
    cusum: float = Field(2.0, gt=0, le=10, description="CUSUM threshold, multiple of daily σ")


def _labels(
    inputs: AnalysisInputs, params: LabelParams
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    close = _prices(inputs)[_field(inputs)].astype(float)
    if len(close) < 500:
        raise InsufficientDataError(f"Labelling needs ≥ 500 bars, got {len(close)}")
    vol = afml.daily_vol(close)
    events = afml.cusum_events(close, vol * params.cusum)
    labels = afml.triple_barrier(close, events, vol, params.pt, params.sl, params.horizon)
    if len(labels) < 50:
        raise InsufficientDataError(
            f"Only {len(labels)} labelled events: lower the CUSUM threshold"
        )
    return close, labels, vol


@analysis(
    id="TR-021",
    version="1.0.0",
    title="Labels & sample weights",
    description=(
        "CUSUM-filtered events (threshold = k × EWM daily σ), triple-barrier labels (profit-take / "
        "stop-loss in σ multiples, vertical barrier), label concurrency and average uniqueness "
        "(returns attributed from the bar after the event — the event day is excluded), effective "
        "sample size Σū, barrier hit shares and class balance."
    ),
    applies_to=PRICE_SUBJECTS,
    params=LabelParams,
    result_kind="table",
    requires={"prices"},
    budget_ms=30_000,
)
def labels_weights(inputs: AnalysisInputs, params: LabelParams) -> AnalysisOutput:
    close, labels, _ = _labels(inputs, params)
    uniq = afml.average_uniqueness(close.index, labels["t1"])
    conc = afml.concurrency(close.index, labels["t1"])
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "events": len(labels),
            "from": labels.index[0].date().isoformat(),
            "to": labels.index[-1].date().isoformat(),
            "barrier_shares": {
                k: float(v) for k, v in labels["barrier"].value_counts(normalize=True).items()
            },
            "class_balance": {
                str(k): float(v) for k, v in labels["label"].value_counts(normalize=True).items()
            },
            "mean_uniqueness": float(uniq.mean()),
            "effective_sample_size": float(uniq.sum()),
            "mean_concurrency": float(conc[conc > 0].mean()),
            "mean_holding_bars": float(((labels["t1"] - labels.index.to_series()).dt.days).mean()),
            "rows": [
                {
                    "t0": t.date().isoformat(),
                    "t1": r["t1"].date().isoformat(),
                    "ret": float(r["ret"]),
                    "label": int(r["label"]),
                    "barrier": r["barrier"],
                    "uniqueness": _f(uniq.get(t)),
                }
                for t, r in labels.tail(30).iterrows()
            ],
        },
    )


class MlParams(LabelParams):
    model: Literal["logistic", "knn"] = "logistic"
    folds: int = Field(5, ge=3, le=10)
    embargo: float = Field(0.01, ge=0, le=0.1)
    cost_bps: float = Field(10.0, ge=0, le=100)


FEATURES = ("ret_5", "ret_20", "ret_60", "vol_20", "rsi_14", "dist_200")


def feature_frame(close: pd.Series) -> pd.DataFrame:
    """Point-in-time features at each bar (trailing windows only)."""
    r = np.log(close).diff()
    return pd.DataFrame(
        {
            "ret_5": r.rolling(5).sum(),
            "ret_20": r.rolling(20).sum(),
            "ret_60": r.rolling(60).sum(),
            "vol_20": r.rolling(20).std(),
            "rsi_14": bt.rsi(close, 14),
            "dist_200": close / close.rolling(200).mean() - 1,
        }
    )


@analysis(
    id="TR-022",
    version="1.0.0",
    title="ML pipeline (purged CV)",
    description=(
        "Features (trailing returns, volatility, RSI, distance to the 200-day average) at each "
        "triple-barrier event → P(profit-take first) by L2 logistic or k-NN, weighted by average "
        "uniqueness; purged K-fold with embargo: log-loss vs naive, precision vs base rate, F1 vs "
        "the trivial classifier, placebo with shuffled labels; MDA and SFI importance; Platt "
        "calibration; bet sizes 2Φ(z) − 1 averaged over active bets → net backtest with costs, "
        "Sharpe and PSR on OOS predictions."
    ),
    applies_to=PRICE_SUBJECTS,
    params=MlParams,
    result_kind="table",
    requires={"prices"},
    budget_ms=60_000,
)
def ml_pipeline(inputs: AnalysisInputs, params: MlParams) -> AnalysisOutput:
    close, labels, _ = _labels(inputs, params)
    features = feature_frame(close).reindex(labels.index)
    ok = features.notna().all(axis=1)
    labels, features = labels.loc[ok], features.loc[ok]
    if len(labels) < 100:
        raise InsufficientDataError(f"Needs ≥ 100 events with full features, got {len(labels)}")
    x = features.to_numpy(dtype=float)
    y = (labels["label"] == 1).astype(float).to_numpy()
    w = (
        afml.average_uniqueness(close.index, labels["t1"])
        .reindex(labels.index)
        .fillna(0.5)
        .to_numpy()
    )
    t0 = labels.index
    splits = afml.purged_kfold(t0, labels["t1"], params.folds, params.embargo)
    model = afml.Logistic if params.model == "logistic" else afml.NearestNeighbours
    scores = afml.cv_scores(model, x, y, splits, w)
    rng = np.random.default_rng(0)
    placebo = afml.cv_scores(model, x, rng.permutation(y), splits, w)
    proba = np.full(len(y), np.nan)
    for train, test in splits:
        proba[test] = model().fit(x[train], y[train], w[train]).proba(x[test])
    good = np.isfinite(proba)
    a, b = afml.platt(proba[good], y[good])
    calibrated = 1 / (1 + np.exp(-(a * np.clip(proba, 1e-6, 1 - 1e-6) + b)))
    sizes = pd.Series(afml.bet_size(np.nan_to_num(calibrated, nan=0.5)), index=t0).clip(
        lower=0
    )  # long-only sizing
    position = afml.average_active_bets(labels["t1"], sizes, close.index)
    frame = bt.run_vectorised(
        close, position, bt.Costs(params.cost_bps / 2, params.cost_bps / 2), warmup=200, lag=0
    )
    net = frame["net"].loc[frame.index >= t0[splits[0][1][0]]].to_numpy()
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "model": params.model,
            "models_pending": PENDING,
            "events": len(y),
            "folds": params.folds,
            "cv": scores,
            "placebo": placebo,
            "beats_placebo_log_loss": bool(scores["log_loss"] < placebo["log_loss"]),
            "mda": afml.mda(x, y, splits, list(FEATURES)),
            "sfi": afml.sfi(x, y, splits, list(FEATURES)),
            "calibration": {"a": a, "b": b},
            "backtest_sharpe": _f(bt.sharpe(net)),
            "backtest_psr": _f(bt.psr(net)),
            "exposure": float(np.mean(position.abs())),
            "note": (
                "Positions are long-only bet sizes averaged over active bets; returns use the "
                "same-bar convention because labels start after the event bar."
            ),
        },
    )


# ── TR-023 forecasting with baselines ────────────────────────────────────────


class ForecastParams(BaseModel):
    target: Literal["returns", "realised_vol"] = "realised_vol"
    horizon: int = Field(5, ge=1, le=63)
    origins: int = Field(60, ge=20, le=250, description="Rolling origins evaluated")
    model: Literal["arima", "ets"] = "ets"


def _series_target(close: pd.Series, target: str) -> pd.Series:
    r = np.log(close).diff().dropna()
    if target == "returns":
        return r
    return (r.rolling(21).std() * math.sqrt(252)).dropna()


def _gauss_crps(y: float, mu: float, sd: float) -> float:
    if sd <= 0:
        return abs(y - mu)
    z = (y - mu) / sd
    return float(
        sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))
    )


@analysis(
    id="TR-023",
    version="1.0.0",
    title="Forecasting vs baselines",
    description=(
        "Rolling-origin evaluation (expanding window, last N origins, horizon h) of a statistical "
        "model (exponential smoothing or ARIMA(1,0,1), statsmodels) against naive, seasonal naive "
        "(5) and historical-mean baselines: MASE vs naive, Gaussian CRPS, 80/95 % interval "
        "coverage, directional accuracy vs 50 % with a binomial test (returns). The baseline "
        "comparison is always shown. Foundation models (Chronos-2, TimesFM) need their dependencies"
        " approved."
    ),
    applies_to=PRICE_SUBJECTS,
    params=ForecastParams,
    result_kind="table",
    requires={"prices"},
    budget_ms=60_000,
)
def forecasting(inputs: AnalysisInputs, params: ForecastParams) -> AnalysisOutput:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    close = _prices(inputs)[_field(inputs)].astype(float)
    y = _series_target(close, params.target).to_numpy()
    h = params.horizon
    if len(y) < 300 + params.origins + h:
        raise InsufficientDataError(
            f"Needs ≥ {300 + params.origins + h} observations, got {len(y)}"
        )
    origins = range(len(y) - h - params.origins, len(y) - h)
    preds: dict[str, list[float]] = {"model": [], "naive": [], "seasonal_naive": [], "mean": []}
    sds, actual, last = [], [], []
    for o in origins:
        hist = y[: o + 1]
        window = hist[-750:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if params.model == "arima":
                    fit = ARIMA(window, order=(1, 0, 1)).fit()
                    fc = fit.get_forecast(h)
                    preds["model"].append(float(fc.predicted_mean[-1]))
                    sds.append(float(np.sqrt(fc.var_pred_mean[-1])))
                else:
                    fit = SimpleExpSmoothing(window, initialization_method="estimated").fit()
                    preds["model"].append(float(fit.forecast(h)[-1]))
                    sds.append(float(np.std(fit.resid, ddof=1) * math.sqrt(h)))
            except (ValueError, np.linalg.LinAlgError):
                preds["model"].append(float(window[-1]))
                sds.append(float(np.std(np.diff(window), ddof=1) * math.sqrt(h)))
        naive = 0.0 if params.target == "returns" else float(hist[-1])
        preds["naive"].append(naive)
        preds["seasonal_naive"].append(float(hist[-5]))
        preds["mean"].append(float(hist.mean()))
        actual.append(float(y[o + h]))
        last.append(float(hist[-1]))
    act = np.array(actual)
    scale = np.mean(np.abs(np.diff(y[: origins[0] + 1]))) or 1e-12
    rows: list[dict[str, Any]] = []
    for name, p in preds.items():
        err = act - np.array(p)
        rows.append(
            {
                "model": name,
                "mae": float(np.abs(err).mean()),
                "mase": float(np.abs(err).mean() / scale),
            }
        )
    sd = np.array(sds)
    pm = np.array(preds["model"])
    cover80 = float((np.abs(act - pm) <= 1.2816 * sd).mean())
    cover95 = float((np.abs(act - pm) <= 1.96 * sd).mean())
    crps = float(np.mean([_gauss_crps(a_, m, s) for a_, m, s in zip(act, pm, sd, strict=True)]))
    direction = None
    if params.target == "returns":
        hits = int(((pm > 0) == (act > 0)).sum())
        direction = {
            "accuracy": hits / len(act),
            "p_value": float(stats.binomtest(hits, len(act), 0.5).pvalue),
        }
    best = min(rows, key=lambda r: r["mase"])
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "target": params.target,
            "horizon": h,
            "origins": params.origins,
            "model": params.model,
            "rows": rows,
            "best": best["model"],
            "model_beats_naive": bool(rows[0]["mase"] < rows[1]["mase"]),
            "crps": crps,
            "coverage_80": cover80,
            "coverage_95": cover95,
            "directional": direction,
            "foundation_models": (
                "Chronos-2 / TimesFM need torch and model weights (dependency awaiting approval)"
            ),
            "expert_note": (
                "Daily return forecasts are usually indistinguishable from naive; volatility "
                "and volume are where gains are plausible."
            ),
        },
    )
