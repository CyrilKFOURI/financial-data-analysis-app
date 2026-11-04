"""Portfolio construction & optimisation (ANALYTICS_CATALOG IN-010…IN-018; module M23).

Universe = the portfolio's holdings (cash excluded), daily base-currency returns over the
context range. Weights are model outputs: they are never applied automatically, and the UI only
offers "save as a new version" once IN-018 (walk-forward validation vs 1/N and other simple
allocators, with costs and a Deflated Sharpe) has been computed for the same configuration.
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core import optimise as opt
from fda_core.analyses.invest import MODEL_LABEL
from fda_core.analyses.portfolio import _assets, _current, _pf, _risk_returns
from fda_core.portfolio import risk_contributions
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

PORTFOLIO: set[Any] = {"portfolio"}
PERIODS = 252

Method = Literal[
    "equal",
    "inverse_vol",
    "gmv",
    "max_sharpe",
    "target_vol",
    "max_utility",
    "erc",
    "hrp",
    "min_cvar",
    "min_cdar",
    "max_diversification",
    "resampled",
    "robust",
]


class View(BaseModel):
    asset: str = Field(max_length=20)
    annual_return: float = Field(ge=-1, le=2)
    confidence: float = Field(0.5, gt=0, lt=1)


class OptParams(BaseModel):
    method: Method = "max_sharpe"
    mean: opt.MeanEstimator = "james_stein"
    cov: opt.CovEstimator = "ledoit_wolf"
    min_weight: float = Field(0.0, ge=-1, le=0.5)
    max_weight: float = Field(1.0, gt=0, le=1)
    rf: float | None = Field(
        None, ge=-0.05, le=0.2, description="Annual; default: currency short rate"
    )
    target_vol: float = Field(0.15, gt=0, le=1, description="Annual (target_vol)")
    risk_aversion: float = Field(4.0, gt=0, le=50)
    alpha: float = Field(0.95, ge=0.8, le=0.995, description="CVaR/CDaR level")
    linkage: Literal["single", "average", "complete", "ward"] = "single"
    views: list[View] = Field(default_factory=list, description="Black–Litterman absolute views")


def _universe(inputs: AnalysisInputs, minimum: int = 120) -> pd.DataFrame:
    r = _risk_returns(inputs, _pf(inputs), minimum).dropna(axis=1, how="all").dropna()
    r = r.loc[:, r.std() > 0]
    if r.shape[1] < 2:
        raise InsufficientDataError("Optimisation needs ≥ 2 holdings with varying returns")
    if len(r) < max(minimum, 3 * r.shape[1]):
        raise InsufficientDataError(
            f"Needs ≥ {max(minimum, 3 * r.shape[1])} common sessions, got {len(r)}"
        )
    return r


def _rf(inputs: AnalysisInputs, params: OptParams) -> float:
    return float(params.rf if params.rf is not None else inputs.conventions.get("rf") or 0.0)


def estimates(returns: pd.DataFrame, params: OptParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-period μ (estimator + Black–Litterman views on top) and Σ."""
    cov = opt.cov_estimate(returns, params.cov)
    mu = opt.mean_estimate(returns, params.mean, cov)
    cols = list(returns.columns)
    views = [
        (cols.index(v.asset.upper()), v.annual_return / PERIODS, v.confidence)
        for v in params.views
        if v.asset.upper() in cols
    ]
    if len(views) < len(params.views):
        missing = [v.asset for v in params.views if v.asset.upper() not in cols]
        raise InsufficientDataError(f"Views on assets not in the universe: {', '.join(missing)}")
    return opt.black_litterman(cov, mu, views), cov


def solve(method: Method, returns: pd.DataFrame, params: OptParams, rf_annual: float) -> np.ndarray:
    mu, cov = estimates(returns, params)
    lo, hi = params.min_weight, params.max_weight
    rf = rf_annual / PERIODS
    if method == "equal":
        return opt.equal_weight(returns.shape[1])
    if method == "inverse_vol":
        return opt.inverse_vol(cov)
    if method == "gmv":
        return opt.gmv(cov, lo, hi)
    if method == "max_sharpe":
        return opt.max_sharpe(mu, cov, rf, lo, hi)
    if method == "target_vol":
        return opt.target_volatility(mu, cov, params.target_vol / math.sqrt(PERIODS), lo, hi)
    if method == "max_utility":
        return opt.max_utility(mu * PERIODS, cov * PERIODS, params.risk_aversion, lo, hi)
    if method == "erc":
        return opt.risk_budget(cov)
    if method == "hrp":
        return opt.hrp(returns, cov, params.linkage)[0]
    if method == "min_cvar":
        return opt.min_cvar(returns.to_numpy(dtype=float), params.alpha, hi)
    if method == "min_cdar":
        return opt.min_cdar(returns.to_numpy(dtype=float), params.alpha, hi)
    if method == "max_diversification":
        return opt.max_diversification(cov, lo, hi)
    if method == "resampled":
        return opt.resampled(
            returns,
            lambda m, c: (
                opt.max_sharpe(m, c, rf, lo, hi) if (m - rf).max() > 0 else opt.gmv(c, lo, hi)
            ),
        )
    robust_mu = opt.robust_mean(mu, cov, len(returns))
    if (robust_mu - rf).max() <= 0:
        return opt.gmv(cov, lo, hi)
    return opt.max_sharpe(robust_mu, cov, rf, lo, hi)


def _weights_payload(
    inputs: AnalysisInputs, returns: pd.DataFrame, w: np.ndarray, params: OptParams, method: str
) -> dict[str, Any]:
    mu, cov = estimates(returns, params)
    current = _current(_pf(inputs))
    sigma, _, rc = risk_contributions(w, cov)
    rf = _rf(inputs, params)
    exp_ret = float(w @ mu) * PERIODS
    vol = sigma * math.sqrt(PERIODS)
    rows = [
        {
            "asset": a,
            "weight": float(w[i]),
            "current": float(current.get(a, 0.0)),
            "delta": float(w[i] - current.get(a, 0.0)),
            "risk_share": float(rc[i] / sigma) if sigma > 0 else None,
        }
        for i, a in enumerate(returns.columns)
    ]
    cur = np.array([float(current.get(a, 0.0)) for a in returns.columns])
    cur_stats = None
    if cur.sum() > 0:
        cur = cur / cur.sum()
        cur_stats = {
            "expected_return": float(cur @ mu) * PERIODS,
            "volatility": float(math.sqrt(cur @ cov @ cur) * math.sqrt(PERIODS)),
        }
    return {
        "label": MODEL_LABEL,
        "method": method,
        "estimators": {"mean": params.mean, "cov": params.cov, "views": len(params.views)},
        "window": [
            returns.index[0].date().isoformat(),
            returns.index[-1].date().isoformat(),
            len(returns),
        ],
        "expected_return": exp_ret,
        "volatility": vol,
        "sharpe": (exp_ret - rf) / vol if vol > 0 else None,
        "turnover_vs_current": float(np.abs(w - cur).sum() / 2) if cur_stats else None,
        "current": cur_stats,
        "validated": False,
        "note": "Run IN-018 (validation) before saving these weights as a version.",
        "rows": sorted(rows, key=lambda r: -r["weight"]),
    }


def _run(inputs: AnalysisInputs, params: OptParams, method: Method) -> dict[str, Any]:
    returns = _universe(inputs)
    w = solve(method, returns, params, _rf(inputs, params))
    return _weights_payload(inputs, returns, w, params, method)


# ── IN-010 … IN-016 ──────────────────────────────────────────────────────────

DESCRIPTION_TAIL = (
    " Model output — weights are proposals; validate with IN-018 before saving a version."
)


@analysis(
    id="IN-010",
    version="1.0.0",
    title="Heuristic portfolios",
    description=(
        "1/N and inverse-volatility weights with ex-ante return, volatility and risk shares "
        "(market-cap weights need share data per holding)." + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=OptParams,
    result_kind="table",
    requires={"portfolio"},
)
def heuristic(inputs: AnalysisInputs, params: OptParams) -> AnalysisOutput:
    method: Method = params.method if params.method in ("equal", "inverse_vol") else "equal"
    return AnalysisOutput("table", _run(inputs, params, method))


@analysis(
    id="IN-011",
    version="1.0.0",
    title="Mean–variance optimiser",
    description=(
        "GMV, max-Sharpe, target volatility or max utility (μ − λ/2·σ²) under weight bounds, by "
        "SLSQP on the chosen estimators (IN-012); efficient frontier with the current and proposed "
        "points; Euler risk shares." + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=OptParams,
    result_kind="table",
    requires={"portfolio"},
    budget_ms=8_000,
)
def mean_variance(inputs: AnalysisInputs, params: OptParams) -> AnalysisOutput:
    method: Method = (
        params.method
        if params.method in ("gmv", "max_sharpe", "target_vol", "max_utility")
        else "max_sharpe"
    )
    returns = _universe(inputs)
    out = _weights_payload(
        inputs, returns, solve(method, returns, params, _rf(inputs, params)), params, method
    )
    mu, cov = estimates(returns, params)
    points = opt.frontier(mu, cov, 15, params.min_weight, params.max_weight)
    out["frontier"] = [
        {"volatility": v * math.sqrt(PERIODS), "expected_return": r * PERIODS} for v, r in points
    ]
    return AnalysisOutput("table", out)


class EstimatorParams(OptParams):
    pass


@analysis(
    id="IN-012",
    version="1.0.0",
    title="Estimators",
    description=(
        "Annualised expected returns per holding by estimator (historical, James–Stein shrinkage, "
        "CAPM equilibrium with equal weights) and the Black–Litterman posterior for the views; "
        "covariance estimators (sample, Ledoit–Wolf, OAS, EWMA, RMT-denoised) with condition "
        "numbers and average correlation."
    ),
    applies_to=PORTFOLIO,
    params=EstimatorParams,
    result_kind="table",
    requires={"portfolio"},
)
def estimator_table(inputs: AnalysisInputs, params: EstimatorParams) -> AnalysisOutput:
    returns = _universe(inputs)
    cols = list(returns.columns)
    base_cov = opt.cov_estimate(returns, params.cov)
    means = {
        m: opt.mean_estimate(returns, m, base_cov) * PERIODS
        for m in ("historical", "james_stein", "equilibrium")
    }
    posterior = estimates(returns, params)[0] * PERIODS
    rows = [
        {
            "asset": a,
            **{m: float(v[i]) for m, v in means.items()},
            "black_litterman": float(posterior[i]),
        }
        for i, a in enumerate(cols)
    ]
    covs = []
    names: tuple[opt.CovEstimator, ...] = ("sample", "ledoit_wolf", "oas", "ewma", "denoised")
    for name in names:
        c = opt.cov_estimate(returns, name)
        sd = np.sqrt(np.diag(c))
        corr = c / np.outer(sd, sd)
        covs.append(
            {
                "estimator": name,
                "condition_number": float(np.linalg.cond(c)),
                "average_correlation": float(corr[np.triu_indices(len(cols), 1)].mean()),
                "average_volatility": float(sd.mean() * math.sqrt(PERIODS)),
            }
        )
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "rows": rows,
            "covariance": covs,
            "views": [v.model_dump() for v in params.views],
        },
    )


class BudgetParams(OptParams):
    budgets: dict[str, float] = Field(
        default_factory=dict, description="asset → risk budget (default equal)"
    )


@analysis(
    id="IN-013",
    version="1.0.0",
    title="Risk parity & budgets",
    description=(
        "Equal risk contribution or custom risk budgets (Spinu convex formulation); "
        "contributions proportional to the budgets." + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=BudgetParams,
    result_kind="table",
    requires={"portfolio"},
)
def risk_parity(inputs: AnalysisInputs, params: BudgetParams) -> AnalysisOutput:
    returns = _universe(inputs)
    _, cov = estimates(returns, params)
    budgets = None
    if params.budgets:
        budgets = np.array([float(params.budgets.get(a, 0.0)) for a in returns.columns])
        if (budgets <= 0).any():
            raise InsufficientDataError("Give a positive risk budget to every holding")
    w = opt.risk_budget(cov, budgets)
    return AnalysisOutput(
        "table",
        _weights_payload(inputs, returns, w, params, "erc" if budgets is None else "risk_budget"),
    )


@analysis(
    id="IN-014",
    version="1.0.0",
    title="Hierarchical risk parity",
    description=(
        "HRP (López de Prado): correlation-distance clustering, quasi-diagonalisation, "
        "recursive bisection with inverse-variance cluster weights; dendrogram linkage and "
        "leaf order." + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=OptParams,
    result_kind="table",
    requires={"portfolio"},
)
def hierarchical(inputs: AnalysisInputs, params: OptParams) -> AnalysisOutput:
    returns = _universe(inputs)
    _, cov = estimates(returns, params)
    w, link = opt.hrp(returns, cov, params.linkage)
    out = _weights_payload(inputs, returns, w, params, "hrp")
    from scipy.cluster import hierarchy

    out["linkage"] = link.tolist()
    out["leaf_order"] = [returns.columns[i] for i in hierarchy.leaves_list(link)]
    return AnalysisOutput("table", out)


@analysis(
    id="IN-015",
    version="1.0.0",
    title="Tail & drawdown optimisers",
    description=(
        "Min-CVaR (Rockafellar–Uryasev LP), min-CDaR (Chekhlov LP) at level α, or max "
        "diversification ratio (Choueifaty); realised CVaR/CDaR of the weights on the sample."
        + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=OptParams,
    result_kind="table",
    requires={"portfolio"},
    budget_ms=15_000,
)
def tail_optimiser(inputs: AnalysisInputs, params: OptParams) -> AnalysisOutput:
    method: Method = (
        params.method
        if params.method in ("min_cvar", "min_cdar", "max_diversification")
        else "min_cvar"
    )
    returns = _universe(inputs)
    w = solve(method, returns, params, _rf(inputs, params))
    out = _weights_payload(inputs, returns, w, params, method)
    r = returns.to_numpy(dtype=float) @ w
    losses = -r
    var = float(np.quantile(losses, params.alpha))
    out["sample_cvar"] = var + float(np.maximum(losses - var, 0).mean() / (1 - params.alpha))
    return AnalysisOutput("table", out)


@analysis(
    id="IN-016",
    version="1.0.0",
    title="Robust & resampled",
    description=(
        "Resampled max-Sharpe (Michaud, 200 bootstraps, averaged) or robust max-Sharpe with "
        "box-uncertainty means μ − σ/√T; weight dispersion vs plain max-Sharpe." + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=OptParams,
    result_kind="table",
    requires={"portfolio"},
    budget_ms=20_000,
)
def robust(inputs: AnalysisInputs, params: OptParams) -> AnalysisOutput:
    method: Method = params.method if params.method in ("resampled", "robust") else "resampled"
    returns = _universe(inputs)
    rf = _rf(inputs, params)
    w = solve(method, returns, params, rf)
    out = _weights_payload(inputs, returns, w, params, method)
    try:
        plain = solve("max_sharpe", returns, params, rf)
        out["distance_to_plain_max_sharpe"] = float(np.abs(w - plain).sum() / 2)
    except InsufficientDataError:
        out["distance_to_plain_max_sharpe"] = None
    return AnalysisOutput("table", out)


class TrackingParams(OptParams):
    names: int = Field(5, ge=1, le=50)
    other: str | None = Field(
        None, max_length=20, description="Benchmark (default: the portfolio's)"
    )


@analysis(
    id="IN-017",
    version="1.0.0",
    title="Index tracking",
    description=(
        "Sparse tracking of the benchmark with N holdings: greedy forward selection on "
        "tracking-error variance, long-only least squares on the selection; annualised "
        "tracking error and R²." + DESCRIPTION_TAIL
    ),
    applies_to=PORTFOLIO,
    params=TrackingParams,
    result_kind="table",
    requires={"portfolio", "other_prices"},
)
def tracking(inputs: AnalysisInputs, params: TrackingParams) -> AnalysisOutput:
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError("Choose a benchmark with stored prices (param `other`)")
    returns = _universe(inputs)
    bench = other["adj_close"].astype(float).pct_change()
    both = returns.join(bench.rename("__bench"), how="inner").dropna()
    x, y = both.drop(columns="__bench").to_numpy(dtype=float), both["__bench"].to_numpy(dtype=float)
    w = opt.index_tracking(x, y, params.names)
    out = _weights_payload(inputs, returns, w, params, "index_tracking")
    resid = x @ w - y
    out["tracking_error"] = float(resid.std(ddof=1) * math.sqrt(PERIODS))
    out["r2"] = float(1 - resid.var() / y.var()) if y.var() > 0 else None
    out["benchmark"] = inputs.conventions.get("other_symbol")
    return AnalysisOutput("table", out)


# ── IN-018 walk-forward validation ───────────────────────────────────────────


DEFAULT_COMPARE: tuple[Method, ...] = ("equal", "inverse_vol", "gmv", "hrp")


class ValidationParams(OptParams):
    lookback: int = Field(756, ge=126, le=2520, description="Estimation window (sessions)")
    cost_bps: float = Field(10.0, ge=0, le=200)
    compare: list[Method] = Field(default_factory=lambda: list(DEFAULT_COMPARE))
    trials: int = Field(0, ge=0, le=1000, description="Configurations tried this session (DSR)")


def walk_forward(
    returns: pd.DataFrame,
    method: Method,
    params: ValidationParams,
    rf: float,
    fixed: np.ndarray | None = None,
) -> tuple[pd.Series, float]:
    """Monthly rebalancing after `lookback` sessions: weights re-estimated on the trailing
    window only (no look-ahead), drifting daily between rebalances, cost = bps × turnover."""
    x = returns.to_numpy(dtype=float)
    dates = returns.index
    month_end = pd.Series(dates.to_period("M"), index=dates)
    rebalance = {
        i
        for i in range(params.lookback, len(dates))
        if i == params.lookback or month_end.iloc[i] != month_end.iloc[i - 1]
    }
    w = np.zeros(x.shape[1])
    daily, turnovers = [], []
    for i in range(params.lookback, len(dates)):
        if i in rebalance:
            if fixed is not None:
                target = fixed
            else:
                window = returns.iloc[i - params.lookback : i]
                try:
                    target = solve(method, window, params, rf)
                except InsufficientDataError:
                    target = w if w.sum() > 0 else opt.equal_weight(x.shape[1])
            turnover = float(np.abs(target - w).sum() / 2) if w.sum() > 0 else 0.0
            turnovers.append(turnover)
            w = target
            cost = turnover * params.cost_bps / 10_000
        else:
            cost = 0.0
        r = float(w @ x[i]) - cost
        daily.append(r)
        grown = w * (1 + x[i])
        w = grown / grown.sum() if grown.sum() > 0 else w
    return pd.Series(daily, index=dates[params.lookback :]), float(
        np.mean(turnovers)
    ) if turnovers else 0.0


def _oos_stats(r: pd.Series, rf: float) -> dict[str, float]:
    nav = (1 + r).cumprod()
    years = len(r) / PERIODS
    vol = float(r.std(ddof=1) * math.sqrt(PERIODS))
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else math.nan
    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": (cagr - rf) / vol if vol > 0 else math.nan,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
    }


@analysis(
    id="IN-018",
    version="1.0.0",
    title="Allocator validation (walk-forward)",
    description=(
        "Walk-forward out-of-sample test: monthly rebalancing, weights re-estimated on the "
        "trailing "
        "`lookback` sessions only, daily drift, costs bps × turnover; the chosen method vs 1/N, "
        "inverse-vol, GMV, HRP and the current weights held constant: CAGR, volatility, Sharpe, "
        "max drawdown, average turnover; Deflated Sharpe of the chosen method over all methods "
        "compared plus the session trial count. Required before saving proposed weights."
    ),
    applies_to=PORTFOLIO,
    params=ValidationParams,
    result_kind="table",
    requires={"portfolio"},
    budget_ms=60_000,
)
def validation(inputs: AnalysisInputs, params: ValidationParams) -> AnalysisOutput:
    pf = _pf(inputs)
    returns = pf.path.returns[_assets(pf)].iloc[1:].dropna()
    returns = returns.loc[:, returns.std() > 0]
    if len(returns) < params.lookback + 126:
        raise InsufficientDataError(
            f"Walk-forward needs ≥ {params.lookback + 126} sessions of holding history, "
            f"got {len(returns)}"
        )
    rf = _rf(inputs, params)
    methods: list[Method] = [params.method, *[m for m in params.compare if m != params.method]]
    series: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for m in methods:
        r, turnover = walk_forward(returns, m, params, rf)
        series[m] = r
        rows.append(
            {
                "method": m,
                "chosen": m == params.method,
                **_oos_stats(r, rf),
                "avg_turnover": turnover,
            }
        )
    current = _current(pf)
    cur = np.array([float(current.get(a, 0.0)) for a in returns.columns])
    if cur.sum() > 0:
        r, turnover = walk_forward(returns, "equal", params, rf, fixed=cur / cur.sum())
        series["current"] = r
        rows.append(
            {
                "method": "current (held)",
                "chosen": False,
                **_oos_stats(r, rf),
                "avg_turnover": turnover,
            }
        )
    chosen = series[params.method]
    excess = chosen - rf / PERIODS
    per_period = [
        float((s - rf / PERIODS).mean() / (s - rf / PERIODS).std(ddof=1)) for s in series.values()
    ]
    from scipy import stats as st

    dsr = opt.deflated_sharpe(
        float(excess.mean() / excess.std(ddof=1)),
        per_period + [0.0] * params.trials,
        len(chosen),
        float(st.skew(excess)),
        float(st.kurtosis(excess, fisher=False)),
    )
    eq = next(r for r in rows if r["method"] == "equal") if "equal" in series else None
    nav = pd.DataFrame({k: (1 + v).cumprod() for k, v in series.items()})
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "validated": True,
            "method": params.method,
            "lookback": params.lookback,
            "cost_bps": params.cost_bps,
            "oos_from": chosen.index[0].date().isoformat(),
            "oos_to": chosen.index[-1].date().isoformat(),
            "rows": rows,
            "deflated_sharpe": dsr["dsr"],
            "dsr_benchmark_sharpe": dsr["sr0"],
            "trials": dsr["trials"],
            "beats_equal_weight": None if eq is None else bool(rows[0]["sharpe"] > eq["sharpe"]),
            "dates": [d.date().isoformat() for d in nav.index[::5]],
            "lines": {k: [float(v) for v in nav[k].to_numpy()[::5]] for k in nav.columns},
        },
    )
