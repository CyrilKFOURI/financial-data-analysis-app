"""Portfolio analyses (ANALYTICS_CATALOG P-001…P-026, C-001…C-007; module M15).

Inputs come from `fda_core.portfolio.PortfolioInputs` (key `portfolio`): the simulated or
replayed path plus holdings metadata. Weights used for "current" views are end-of-day weights at
the last close; risk and interaction statistics use the context window of holding returns in
the base currency. Every price/risk/time-series analysis also runs on the portfolio's TWR NAV
through the Subject abstraction (PRICE_SUBJECTS includes "portfolio").
"""

import itertools
import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from fda_core.analyses.peers import SECTOR_ETF
from fda_core.analyses.price import window_start
from fda_core.analyses.risk import STRESS_WINDOWS
from fda_core.portfolio import CASH, PortfolioInputs, covariance, risk_contributions, xirr
from fda_core.registry import (
    AnalysisInputs,
    AnalysisOutput,
    InsufficientDataError,
    NotApplicableError,
    analysis,
)

PORTFOLIO: set[Any] = {"portfolio"}
NEEDS = {"portfolio"}


class NoParams(BaseModel):
    pass


def _pf(inputs: AnalysisInputs) -> PortfolioInputs:
    pf = inputs.data.get("portfolio")
    if not isinstance(pf, PortfolioInputs):
        raise InsufficientDataError("Portfolio definition not loaded")
    return pf


def _factor(inputs: AnalysisInputs) -> int:
    return int(inputs.conventions.get("annualisation") or 252)


def _window(inputs: AnalysisInputs, frame: pd.DataFrame | pd.Series) -> Any:
    if frame.empty:
        return frame
    code = str(inputs.conventions.get("range", "1Y"))
    start = window_start(code, frame.index[-1].date(), frame.index[0].date())
    return frame.loc[frame.index >= pd.Timestamp(start)]


def _iso(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _assets(pf: PortfolioInputs) -> list[str]:
    return [c for c in pf.path.returns.columns if c != CASH]


def _risk_returns(inputs: AnalysisInputs, pf: PortfolioInputs, minimum: int = 30) -> pd.DataFrame:
    """Window of holdings' base-currency returns (cash excluded), from each holding's first day."""
    r = _window(inputs, pf.path.returns[_assets(pf)].iloc[1:])
    if len(r) < minimum:
        raise InsufficientDataError(f"Needs ≥ {minimum} sessions of holding returns, got {len(r)}")
    return r


def _current(pf: PortfolioInputs) -> pd.Series:
    w = pf.end_weights()
    return w[w.abs() > 1e-12]


# ── P-001 weights mode / P-002 ledger / P-003 TWR / P-004 MWR ────────────────


@analysis(
    id="P-001",
    version="1.0.0",
    title="Weights & drift",
    description=(
        "Target vs current weights (drift since the last rebalance), the rebalancing assumption "
        "(D-PF-2), number of rebalances and costs paid over the window."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def weights_mode(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    current = pf.end_weights()
    targets = {
        ("CASH" if h.get("is_cash") else h["symbol"]): float(h["weight"])
        for h in pf.definition.get("holdings") or []
    }
    rows = [
        {
            "symbol": s,
            "target": targets.get(s),
            "current": _f(current.get(s, 0.0)),
            "drift": None if s not in targets else _f(current.get(s, 0.0) - targets[s]),
        }
        for s in sorted(
            set(current.index) | set(targets), key=lambda s: -float(current.get(s, 0.0))
        )
    ]
    turnover = _window(inputs, pf.path.turnover)
    return AnalysisOutput(
        "table",
        {
            "mode": pf.definition.get("mode"),
            "rows": rows,
            "rebalance": pf.definition.get("rebalance"),
            "cost_bps": pf.definition.get("cost_bps"),
            "rebalances": int((turnover > 0).sum()),
            "costs_paid": _f(_window(inputs, pf.path.costs).sum()),
        },
        tuple(pf.path.warnings),
    )


@analysis(
    id="P-002",
    version="1.0.0",
    title="Ledger & cash",
    description=(
        "Transactions (most recent first), counts by type and the base-currency cash balance."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def ledger(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    tx = pf.definition.get("transactions")
    if not isinstance(tx, pd.DataFrame):
        raise NotApplicableError("Weights-mode portfolio: no ledger (convert it to transactions)")
    rows = tx.sort_values("trade_date", ascending=False).head(500)
    records = [
        {
            "date": pd.Timestamp(r.trade_date).date().isoformat(),
            "type": r.type,
            "symbol": r.symbol,
            "quantity": _f(r.quantity),
            "price": _f(r.price),
            "currency": r.currency,
            "fees": _f(getattr(r, "fees", 0.0)),
        }
        for r in rows.itertuples(index=False)
    ]
    return AnalysisOutput(
        "table",
        {
            "transactions": records,
            "counts": tx["type"].value_counts().to_dict(),
            "cash": _f(pf.path.cash.iloc[-1]) if pf.path.cash is not None else None,
        },
        tuple(pf.path.warnings),
    )


def _ann(total: float, days: float) -> float | None:
    return (1 + total) ** (365.25 / days) - 1 if days >= 365 and total > -1 else None


@analysis(
    id="P-003",
    version="1.0.0",
    title="Time-weighted return",
    description=(
        "Daily chain-linked TWR ∏ (V_t − CF_t)/V_{t−1} − 1 (external flows removed): window, "
        "annualised (≥ 1 year), YTD and calendar years, with the NAV."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="series",
    requires=NEEDS,
)
def twr(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    nav = pf.path.nav
    w = _window(inputs, nav)
    total = float(w.iloc[-1] / w.iloc[0] - 1)
    days = (w.index[-1] - w.index[0]).days
    year_end = nav.groupby(nav.index.year).last()
    years = (year_end / year_end.shift(1).fillna(nav.iloc[0]) - 1).to_dict()
    ytd_base = nav.loc[nav.index.year < nav.index[-1].year]
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(w.index),
            "nav": (100 * w / w.iloc[0]).tolist(),
            "twr": total,
            "twr_annualised": _ann(total, days),
            "inception_twr": float(nav.iloc[-1] / nav.iloc[0] - 1),
            "ytd": float(nav.iloc[-1] / ytd_base.iloc[-1] - 1) if len(ytd_base) else None,
            "calendar_years": [{"year": int(y), "twr": float(v)} for y, v in years.items()],
        },
    )


@analysis(
    id="P-004",
    version="1.0.0",
    title="Money-weighted return (XIRR)",
    description=(
        "IRR of the dated external flows plus the terminal value since inception, next to the "
        "annualised TWR: the gap measures the effect of the timing of deposits and withdrawals."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="value",
    requires=NEEDS,
)
def mwr(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    path = pf.path
    flows = path.flows[path.flows != 0]
    if flows.empty:  # weights mode: one initial investment
        dates, amounts = [path.value.index[0]], [-float(path.value.iloc[0])]
    else:
        dates, amounts = list(flows.index), [-float(v) for v in flows]
    dates.append(path.value.index[-1])
    amounts.append(float(path.value.iloc[-1]))
    rate = xirr(dates, amounts)
    days = (path.nav.index[-1] - path.nav.index[0]).days
    total = float(path.nav.iloc[-1] / path.nav.iloc[0] - 1)
    twr_ann = (1 + total) ** (
        365.0 / days
    ) - 1  # same day count as XIRR if days > 0 and total > -1 else None
    gap = None if rate is None or twr_ann is None else rate - twr_ann
    reading = (
        None
        if gap is None
        else "Money was added before stronger periods (timing helped)"
        if gap > 0.005
        else "Money was added before weaker periods (timing hurt)"
        if gap < -0.005
        else "Flows timing had little effect"
    )
    return AnalysisOutput(
        "value",
        {
            "mwr": rate,
            "twr_annualised": twr_ann,
            "gap": gap,
            "reading": reading,
            "external_flows": len(flows),
            "net_flows": float(flows.sum()),
            "since": _iso(path.nav.index[:1])[0],
        },
    )


# ── P-005 positions / P-006 allocations / P-007 concentration ────────────────


@analysis(
    id="P-005",
    version="1.0.0",
    title="Positions & P&L",
    description=(
        "Per holding: quantity, average cost of open lots (FIFO/LIFO/average per D-PF-3), "
        "market value, weight, day change, unrealised and realised P&L, holding period (days "
        "since the oldest open lot) and asset type."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def positions(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    path = pf.path
    hv = path.holdings_value
    if hv is None:
        raise InsufficientDataError("No holdings values")
    last, prev = hv.iloc[-1], hv.iloc[-2] if len(hv) > 1 else hv.iloc[-1]
    total = float(last.sum())
    day = path.returns.iloc[-1]
    realised: dict[str, float] = {}
    for r in path.realised:
        realised[r["symbol"]] = realised.get(r["symbol"], 0.0) + r["pnl"]
    rows: list[dict[str, Any]] = []
    for s in hv.columns:
        mv = float(last[s])
        if abs(mv) < 1e-9 and s not in realised:
            continue
        lots = [lot for lot in path.lots if lot["symbol"] == s]
        qty = (
            float(path.quantities[s].iloc[-1])
            if path.quantities is not None and s in path.quantities
            else None
        )
        cost = sum(lot["quantity"] * lot["unit_cost"] for lot in lots) if lots else None
        oldest = min((lot["opened"] for lot in lots), default=None)
        rows.append(
            {
                "symbol": s,
                "type": (pf.meta.get(s) or {}).get("asset_class", "cash" if s == CASH else None),
                "quantity": qty,
                "avg_cost": cost / qty if cost is not None and qty else None,
                "market_value": mv,
                "weight": mv / total if total else None,
                "day_change": float(prev[s]) * float(day.get(s, 0.0)) if s != CASH else 0.0,
                "unrealised": mv - cost if cost is not None else None,
                "realised": realised.get(s),
                "holding_days": (path.value.index[-1] - pd.Timestamp(oldest)).days
                if oldest
                else None,
            }
        )
    rows.sort(key=lambda r: -abs(r["market_value"]))
    return AnalysisOutput(
        "table",
        {
            "positions": rows,
            "value": total,
            "realised_total": sum(realised.values()),
            "unrealised_total": sum(r["unrealised"] or 0.0 for r in rows),
            "income_total": sum(i["amount"] for i in path.income),
        },
        tuple(path.warnings),
    )


REGIONS = {
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "Latin America",
    "Brazil": "Latin America",
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Netherlands": "Europe",
    "Switzerland": "Europe",
    "Ireland": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Denmark": "Europe",
    "Belgium": "Europe",
    "Finland": "Europe",
    "Norway": "Europe",
    "Japan": "Asia-Pacific",
    "China": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific",
    "Taiwan": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "India": "Asia-Pacific",
    "Australia": "Asia-Pacific",
    "Singapore": "Asia-Pacific",
    "Israel": "Middle East",
}


def cap_bucket(cap: float | None) -> str:
    if cap is None:
        return "n/a"
    for floor, label in ((200e9, "mega"), (10e9, "large"), (2e9, "mid"), (300e6, "small")):
        if cap >= floor:
            return label
    return "micro"


# SPDR sector ETFs stand for their sector (sector-ETF benchmarks for Brinson, allocations).
ETF_SECTOR = {etf: key.replace("-", " ").title() for key, etf in SECTOR_ETF.items()}


def _dimension(pf: PortfolioInputs, symbol: str, dim: str) -> str:
    if symbol == CASH:
        return "Cash"
    if dim == "sector" and symbol in ETF_SECTOR:
        return ETF_SECTOR[symbol]
    meta = pf.meta.get(symbol) or {}
    info = meta.get("info") or {}
    if dim == "region":
        return REGIONS.get(str(meta.get("country")), "Other" if meta.get("country") else "n/a")
    if dim == "market_cap":
        return cap_bucket(_f(info.get("marketCap")))
    return str(meta.get(dim) or "n/a")


DIMENSIONS = (
    "asset_class",
    "asset_subclass",
    "sector",
    "industry",
    "country",
    "region",
    "currency",
    "market_cap",
)


@analysis(
    id="P-006",
    version="1.0.0",
    title="Allocation breakdowns",
    description=(
        "Current weights grouped by asset class, subclass, sector, industry, country, region, "
        "listing currency and market-cap bucket (mega ≥ 200 bn, large ≥ 10 bn, mid ≥ 2 bn, "
        "small ≥ 300 m)."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def allocations(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    w = _current(pf)
    out = {}
    for dim in DIMENSIONS:
        groups: dict[str, float] = {}
        for s, v in w.items():
            key = _dimension(pf, str(s), dim)
            groups[key] = groups.get(key, 0.0) + float(v)
        out[dim] = [
            {"group": k, "weight": v} for k, v in sorted(groups.items(), key=lambda kv: -kv[1])
        ]
    return AnalysisOutput("table", {"breakdowns": out})


@analysis(
    id="P-007",
    version="1.0.0",
    title="Concentration",
    description=(
        "Max weight, top-5 weight, Herfindahl index HHI = Σ w² and effective N = 1/HHI (cash "
        "included)."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="value",
    requires=NEEDS,
)
def concentration(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    w = _current(_pf(inputs)).abs().sort_values(ascending=False)
    hhi = float((w**2).sum())
    return AnalysisOutput(
        "value",
        {
            "max_weight": float(w.iloc[0]),
            "max_holding": str(w.index[0]),
            "top5": float(w.iloc[:5].sum()),
            "hhi": hhi,
            "effective_n": 1 / hhi if hhi else None,
            "holdings": len(w),
        },
    )


# ── P-008 contribution ───────────────────────────────────────────────────────


@analysis(
    id="P-008",
    version="1.0.0",
    title="Contribution to return",
    description=(
        "Σ_t w_{i,t−1} r_{i,t} per holding over the window (arithmetic, sums to the portfolio's "
        "simple-return sum), by calendar year, with top contributors and detractors."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def contribution(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    c = _window(inputs, (pf.path.weights * pf.path.returns).iloc[1:])
    if c.empty:
        raise InsufficientDataError("No returns in the window")
    total = c.sum().sort_values(ascending=False)
    by_year = c.groupby(c.index.year).sum()
    cumulative = c.cumsum()
    top = list(total.abs().sort_values(ascending=False).index[:6])
    nav = _window(inputs, pf.path.nav)
    return AnalysisOutput(
        "table",
        {
            "rows": [{"symbol": s, "contribution": float(v)} for s, v in total.items()],
            "sum": float(total.sum()),
            "twr": float(nav.iloc[-1] / nav.iloc[0] - 1),
            "by_year": [
                {"year": int(y)} | {s: float(v) for s, v in row.items()}
                for y, row in by_year.iterrows()
            ],
            "top": [s for s in total.index[:5] if total[s] > 0],
            "bottom": [s for s in total.index[::-1][:5] if total[s] < 0],
            "dates": _iso(cumulative.index),
            "cumulative": {s: cumulative[s].tolist() for s in top},
        },
    )


# ── P-009…P-013 risk decomposition ───────────────────────────────────────────


class RiskParams(BaseModel):
    estimator: Literal["sample", "ledoit_wolf", "oas", "ewma"] = "sample"


def _decompose(inputs: AnalysisInputs, pf: PortfolioInputs, estimator: str) -> dict[str, Any]:
    r = _risk_returns(inputs, pf)
    w_all = _current(pf)
    names = [s for s in r.columns if s in w_all.index]
    w = w_all.reindex(names).to_numpy()
    cov = covariance(r[names], estimator) * _factor(inputs)  # type: ignore[arg-type]
    sigma, marginal, contrib = risk_contributions(w, cov)
    vols = np.sqrt(np.diag(cov))
    return {
        "names": names,
        "w": w,
        "cov": cov,
        "sigma": sigma,
        "marginal": marginal,
        "contrib": contrib,
        "vols": vols,
        "cash": float(w_all.get(CASH, 0.0)),
    }


@analysis(
    id="P-009",
    version="1.0.0",
    title="Risk decomposition",
    description=(
        "Euler decomposition of annualised volatility: RC_i = w_i (Σw)_i / σ_p (sums to σ_p), "
        "marginal contributions and risk share vs weight share, with the chosen covariance "
        "estimator (P-012) over the window."
    ),
    applies_to=PORTFOLIO,
    params=RiskParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=30,
)
def risk_decomposition(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    d = _decompose(inputs, _pf(inputs), params.estimator)
    rows = [
        {
            "symbol": s,
            "weight": float(w),
            "volatility": float(v),
            "marginal": float(m),
            "risk_contribution": float(c),
            "risk_share": float(c / d["sigma"]) if d["sigma"] else None,
        }
        for s, w, v, m, c in zip(
            d["names"], d["w"], d["vols"], d["marginal"], d["contrib"], strict=True
        )
    ]
    rows.sort(key=lambda r: -r["risk_contribution"])
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "volatility": d["sigma"],
            "estimator": params.estimator,
            "cash_weight": d["cash"],
            "check_sum": float(d["contrib"].sum()),
        },
    )


class EsParams(BaseModel):
    confidence: float = Field(0.95, ge=0.8, le=0.999)


@analysis(
    id="P-010",
    version="1.0.0",
    title="Component VaR / ES",
    description=(
        "Historical simulation with current weights over the window: VaR and ES at the "
        "confidence level; component ES_i = −mean(w_i r_i | portfolio return ≤ −VaR), which sums "
        "to ES."
    ),
    applies_to=PORTFOLIO,
    params=EsParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=60,
)
def component_es(inputs: AnalysisInputs, params: EsParams) -> AnalysisOutput:
    pf = _pf(inputs)
    r = _risk_returns(inputs, pf, 60)
    w = _current(pf).reindex(r.columns).fillna(0.0)
    pnl = r.mul(w, axis=1)
    rp = pnl.sum(axis=1)
    var = float(-np.quantile(rp, 1 - params.confidence))
    tail = rp <= -var
    comp = -pnl.loc[tail].mean()
    es = float(-rp.loc[tail].mean())
    rows = [
        {
            "symbol": s,
            "weight": float(w[s]),
            "component_es": float(v),
            "share": float(v / es) if es else None,
        }
        for s, v in comp.sort_values(ascending=False).items()
    ]
    return AnalysisOutput(
        "table",
        {
            "var": var,
            "es": es,
            "confidence": params.confidence,
            "tail_days": int(tail.sum()),
            "rows": rows,
        },
    )


@analysis(
    id="P-011",
    version="1.0.0",
    title="Diversification ratio",
    description=(
        "DR = Σ w_i σ_i / σ_p (≥ 1; 1 = no diversification) on the window, with the equal-weight "
        "DR for reference."
    ),
    applies_to=PORTFOLIO,
    params=RiskParams,
    result_kind="value",
    requires=NEEDS,
    min_obs=30,
)
def diversification_ratio(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    d = _decompose(inputs, _pf(inputs), params.estimator)
    dr = float(d["w"] @ d["vols"] / d["sigma"]) if d["sigma"] else None
    n = len(d["w"])
    eq = np.full(n, 1 / n)
    sigma_eq = math.sqrt(float(eq @ d["cov"] @ eq))
    return AnalysisOutput(
        "value",
        {
            "diversification_ratio": dr,
            "equal_weight_dr": float(eq @ d["vols"] / sigma_eq) if sigma_eq else None,
            "weighted_volatility": float(d["w"] @ d["vols"]),
            "portfolio_volatility": d["sigma"],
        },
    )


@analysis(
    id="P-012",
    version="1.0.0",
    title="Covariance estimators",
    description=(
        "Portfolio volatility under sample, Ledoit–Wolf (constant-variance target), OAS and "
        "EWMA (half-life 63) covariance, with each matrix's condition number."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=30,
)
def covariance_estimators(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    rows: list[dict[str, Any]] = []
    for est in ("sample", "ledoit_wolf", "oas", "ewma"):
        d = _decompose(inputs, pf, est)
        eig = np.linalg.eigvalsh(d["cov"])
        rows.append(
            {
                "estimator": est,
                "volatility": d["sigma"],
                "condition_number": float(eig[-1] / eig[0]) if eig[0] > 0 else None,
            }
        )
    return AnalysisOutput("table", {"rows": rows, "assets": len(_assets(pf))})


class AbsorptionParams(BaseModel):
    window: int = Field(252, ge=60, le=756)


@analysis(
    id="P-013",
    version="1.0.0",
    title="Absorption ratio",
    description=(
        "Share of total variance explained by the top k = ⌈n/5⌉ principal components (and PC1) "
        "of the holdings' return covariance, on a rolling window sampled every 21 sessions."
    ),
    applies_to=PORTFOLIO,
    params=AbsorptionParams,
    result_kind="series",
    requires=NEEDS,
    min_obs=60,
)
def absorption_ratio(inputs: AnalysisInputs, params: AbsorptionParams) -> AnalysisOutput:
    pf = _pf(inputs)
    r = pf.path.returns[_assets(pf)].iloc[1:]
    n = r.shape[1]
    if n < 3:
        raise InsufficientDataError("Absorption ratio needs ≥ 3 holdings")
    k = math.ceil(n / 5)
    dates, ar, pc1 = [], [], []
    for end in range(params.window, len(r) + 1, 21):
        eig = np.linalg.eigvalsh(
            np.cov(r.iloc[end - params.window : end].to_numpy(), rowvar=False)
        )[::-1]
        dates.append(r.index[end - 1])
        ar.append(float(eig[:k].sum() / eig.sum()))
        pc1.append(float(eig[0] / eig.sum()))
    if not dates:
        raise InsufficientDataError(f"Needs ≥ {params.window} sessions")
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(pd.DatetimeIndex(dates)),
            "absorption": ar,
            "pc1_share": pc1,
            "k": k,
            "current": ar[-1],
        },
    )


@analysis(
    id="P-017",
    version="1.0.0",
    title="Turnover & costs",
    description=(
        "Annualised one-way turnover and cost drag (bps a year of average value) under the "
        "rebalancing assumption, or from the ledger's traded notional in transactions mode."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="value",
    requires=NEEDS,
)
def turnover_costs(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    path = pf.path
    value = _window(inputs, path.value)
    years = max((value.index[-1] - value.index[0]).days / 365.25, 1 / 252)
    if path.quantities is not None and path.close_base is not None:
        q = _window(inputs, path.quantities)
        traded = (
            (q.diff().abs() * _window(inputs, path.close_base).reindex(q.index))
            .sum(axis=1)
            .iloc[1:]
        )
        turnover = float(traded.sum() / 2 / value.mean())
        fees = float(
            pd.to_numeric(
                pf.definition["transactions"].get("fees", pd.Series(dtype=float)), errors="coerce"
            ).sum()
        )
    else:
        turnover = float(_window(inputs, path.turnover).sum())
        fees = float(_window(inputs, path.costs).sum())
    return AnalysisOutput(
        "value",
        {
            "turnover_annual": turnover / years,
            "costs": fees,
            "cost_drag_bps_annual": 1e4 * fees / value.mean() / years,
            "rebalances": int((_window(inputs, path.turnover) > 0).sum()),
        },
    )


# ── P-018 Brinson / P-025 active share / P-026 overlap (vs another portfolio) ─


class CompareParams(BaseModel):
    compare: int | None = Field(
        None, ge=1, description="Portfolio id (default: the benchmark portfolio)"
    )


def _compare(inputs: AnalysisInputs) -> PortfolioInputs:
    other = inputs.data.get("compare_portfolio")
    if not isinstance(other, PortfolioInputs):
        raise InsufficientDataError(
            "Needs a benchmark portfolio: set the portfolio's benchmark to a portfolio or pick"
            " one to compare"
        )
    return other


def _monthly(pf: PortfolioInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Start-of-month weights and compounded monthly holding returns."""
    r = pf.path.returns.iloc[1:]
    months = r.index.to_period("M")
    monthly = (1 + r).groupby(months).prod() - 1
    weights = pf.path.weights.iloc[1:].groupby(months).first()
    return weights, monthly


def _sector_view(
    pf: PortfolioInputs, w: pd.Series, r: pd.Series
) -> tuple[dict[str, float], dict[str, float]]:
    wsec: dict[str, float] = {}
    csec: dict[str, float] = {}
    for s in w.index:
        key = _dimension(pf, str(s), "sector")
        wsec[key] = wsec.get(key, 0.0) + float(w[s])
        csec[key] = csec.get(key, 0.0) + float(w[s]) * float(r.get(s, 0.0))
    return wsec, {k: csec[k] / wsec[k] for k in wsec if abs(wsec[k]) > 1e-12}


def brinson_fachler(
    wp: dict[str, float], rp: dict[str, float], wb: dict[str, float], rb: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Single-period Brinson–Fachler effects by sector (sum = R_p − R_b)."""
    total_b = sum(wb[k] * rb.get(k, 0.0) for k in wb)
    out = {}
    for k in sorted(set(wp) | set(wb)):
        w, b = wp.get(k, 0.0), wb.get(k, 0.0)
        bk = rb.get(k, total_b) if b else total_b
        rk = rp.get(k, bk) if w else bk
        out[k] = {
            "allocation": (w - b) * (bk - total_b),
            "selection": b * (rk - bk),
            "interaction": (w - b) * (rk - bk),
        }
    return out


def _carino(r: float, b: float) -> float:
    return 1.0 if abs(r - b) < 1e-12 else (math.log1p(r) - math.log1p(b)) / (r - b)


@analysis(
    id="P-018",
    version="1.0.0",
    title="Brinson attribution",
    description=(
        "Brinson–Fachler allocation, selection and interaction by sector vs a benchmark "
        "portfolio, per month, linked over the window with Carino factors (sum = geometric "
        "excess return)."
    ),
    applies_to=PORTFOLIO,
    params=CompareParams,
    result_kind="table",
    requires=NEEDS | {"compare_portfolio"},
)
def brinson(inputs: AnalysisInputs, params: CompareParams) -> AnalysisOutput:
    pf, bm = _pf(inputs), _compare(inputs)
    wp_m, rp_m = _monthly(pf)
    wb_m, rb_m = _monthly(bm)
    months = _window(
        inputs, pd.Series(0, index=wp_m.index.to_timestamp(how="end").normalize())
    ).index.to_period("M")
    months = [m for m in months if m in wb_m.index]
    if not months:
        raise InsufficientDataError("No overlapping months with the benchmark portfolio")
    effects: list[tuple[float, float, dict[str, dict[str, float]]]] = []
    for m in months:
        wsp, rsp = _sector_view(pf, wp_m.loc[m], rp_m.loc[m])
        wsb, rsb = _sector_view(bm, wb_m.loc[m], rb_m.loc[m])
        r_p = sum(wsp[k] * rsp.get(k, 0.0) for k in wsp)
        r_b = sum(wsb[k] * rsb.get(k, 0.0) for k in wsb)
        effects.append((r_p, r_b, brinson_fachler(wsp, rsp, wsb, rsb)))
    total_p = float(np.prod([1 + e[0] for e in effects]) - 1)
    total_b = float(np.prod([1 + e[1] for e in effects]) - 1)
    k_total = _carino(total_p, total_b)
    linked: dict[str, dict[str, float]] = {}
    periods = []
    for m, (r_p, r_b, eff) in zip(months, effects, strict=True):
        k = _carino(r_p, r_b) / k_total
        periods.append({"month": str(m), "portfolio": r_p, "benchmark": r_b, "excess": r_p - r_b})
        for sector, e in eff.items():
            acc = linked.setdefault(
                sector, {"allocation": 0.0, "selection": 0.0, "interaction": 0.0}
            )
            for name in acc:
                acc[name] += e[name] * k
    rows: list[dict[str, Any]] = [
        {"sector": s} | v | {"total": sum(v.values())} for s, v in linked.items()
    ]
    rows.sort(key=lambda r: -abs(r["total"]))
    unknown = sum(
        float(v) for s, v in _current(bm).items() if _dimension(bm, str(s), "sector") == "n/a"
    )
    warnings = (
        (
            f"{unknown:.0%} of the benchmark has no sector (funds): use stocks or SPDR sector ETFs "
            "in the benchmark portfolio for a meaningful sector attribution.",
        )
        if unknown > 0.2
        else ()
    )
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "portfolio_return": total_p,
            "benchmark_return": total_b,
            "excess": total_p - total_b,
            "benchmark": bm.definition.get("name"),
            "periods": periods[-24:],
        },
        warnings,
    )


@analysis(
    id="P-025",
    version="1.0.0",
    title="Active share & tracking error",
    description=(
        "Active share = ½ Σ |w_p − w_b| over the union of holdings vs a benchmark portfolio, and "
        "ex-ante tracking error √(aᵀΣa) of the active weights (sample covariance, window)."
    ),
    applies_to=PORTFOLIO,
    params=CompareParams,
    result_kind="value",
    requires=NEEDS | {"compare_portfolio"},
)
def active_share(inputs: AnalysisInputs, params: CompareParams) -> AnalysisOutput:
    pf, bm = _pf(inputs), _compare(inputs)
    wp, wb = _current(pf), _current(bm)
    names = sorted(set(wp.index) | set(wb.index))
    a = wp.reindex(names).fillna(0.0) - wb.reindex(names).fillna(0.0)
    returns = pd.concat([pf.path.returns, bm.path.returns], axis=1)
    returns = returns.loc[:, ~returns.columns.duplicated()].reindex(columns=names).fillna(0.0)
    r = _window(inputs, returns.iloc[1:])
    cov = np.cov(r.to_numpy(), rowvar=False) * _factor(inputs)
    te = math.sqrt(max(float(a.to_numpy() @ cov @ a.to_numpy()), 0.0))
    return AnalysisOutput(
        "value",
        {
            "active_share": float(a.abs().sum() / 2),
            "tracking_error_ex_ante": te,
            "benchmark": bm.definition.get("name"),
            "largest_active": [
                {"symbol": s, "active_weight": float(v)}
                for s, v in a.sort_values(key=abs, ascending=False).head(10).items()
            ],
        },
    )


@analysis(
    id="P-026",
    version="1.0.0",
    title="Holdings overlap",
    description=(
        "Overlap = Σ min(w_a, w_b) between this portfolio and another one, with the common "
        "holdings."
    ),
    applies_to=PORTFOLIO,
    params=CompareParams,
    result_kind="table",
    requires=NEEDS | {"compare_portfolio"},
)
def overlap(inputs: AnalysisInputs, params: CompareParams) -> AnalysisOutput:
    pf, other = _pf(inputs), _compare(inputs)
    wa, wb = _current(pf).drop(CASH, errors="ignore"), _current(other).drop(CASH, errors="ignore")
    common = sorted(set(wa.index) & set(wb.index), key=lambda s: -min(wa[s], wb[s]))
    return AnalysisOutput(
        "table",
        {
            "overlap": float(sum(min(wa[s], wb[s]) for s in common)),
            "other": other.definition.get("name"),
            "common": [
                {"symbol": s, "weight": float(wa[s]), "other_weight": float(wb[s])} for s in common
            ],
        },
    )


# ── P-019 look-through / P-020 currency / P-021 liquidity ────────────────────

LOOK_THROUGH = {
    "pe": ("trailingPE", "harmonic"),
    "forward_pe": ("forwardPE", "harmonic"),
    "ev_ebitda": ("enterpriseToEbitda", "harmonic"),
    "price_to_book": ("priceToBook", "harmonic"),
    "roe": ("returnOnEquity", "mean"),
    "operating_margin": ("operatingMargins", "mean"),
    "gross_margin": ("grossMargins", "mean"),
    "revenue_growth": ("revenueGrowth", "mean"),
    "earnings_growth": ("earningsGrowth", "mean"),
    "debt_to_equity": ("debtToEquity", "mean"),
    "dividend_yield": ("dividendYield", "mean"),
}


@analysis(
    id="P-019",
    version="1.0.0",
    title="Look-through fundamentals",
    description=(
        "Portfolio-level multiples as weighted harmonic means (1 / Σ wᵢ/Mᵢ over covered weight, "
        "negative earnings included as negative yields) and weighted means for margins, returns "
        "and growth, over company holdings only (D-PF-7), with the covered weight per metric."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def look_through(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    w = _current(pf)
    companies = {
        s: float(v)
        for s, v in w.items()
        if (pf.meta.get(str(s)) or {}).get("asset_class") == "equity"
        and (pf.meta.get(str(s)) or {}).get("asset_subclass") not in ("etf", "fund", "index")
    }
    if not companies:
        raise InsufficientDataError("No company holdings: look-through applies to single stocks")
    rows: list[dict[str, Any]] = []
    for metric, (key, how) in LOOK_THROUGH.items():
        pairs = [
            (v, _f(((pf.meta.get(s) or {}).get("info") or {}).get(key)))
            for s, v in companies.items()
        ]
        valid = [(v, x) for v, x in pairs if x is not None and (how == "mean" or x != 0)]
        covered = sum(v for v, _ in valid)
        if not valid or covered <= 0:
            rows.append({"metric": metric, "value": None, "covered_weight": 0.0})
            continue
        if how == "harmonic":
            inv = sum(v / x for v, x in valid) / covered
            value = 1 / inv if inv else None
        else:
            value = sum(v * x for v, x in valid) / covered
        rows.append({"metric": metric, "value": value, "covered_weight": covered})
    uncovered = sorted(s for s in companies if not (pf.meta.get(s) or {}).get("info"))
    notes = ["Fundamentals are the provider's latest snapshot per holding (not point-in-time)."]
    if uncovered:
        notes.append(f"No company data yet for {', '.join(uncovered)}: ingest it to cover them.")
    return AnalysisOutput(
        "table",
        {"rows": rows, "company_weight": sum(companies.values()), "uncovered": uncovered},
        tuple(notes),
    )


@analysis(
    id="P-020",
    version="1.0.0",
    title="Currency exposure",
    description=(
        "Current weights by listing currency and, over the window, the FX contribution "
        "Σ_t w_{i,t−1}(r^base_{i,t} − r^local_{i,t}) per currency next to the local-return "
        "contribution."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def currency_exposure(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    base = pf.definition.get("base_currency")
    ccy = {
        s: (base if s == CASH else (pf.meta.get(s) or {}).get("currency") or base)
        for s in pf.path.returns.columns
    }
    w = _current(pf)
    weights = pf.path.weights.iloc[1:]
    fx_part = _window(inputs, weights * (pf.path.returns - pf.path.local_returns).iloc[1:])
    local_part = _window(inputs, weights * pf.path.local_returns.iloc[1:])
    rows: list[dict[str, Any]] = []
    for c in sorted({str(v) for v in ccy.values()}):
        members = [s for s, v in ccy.items() if v == c]
        rows.append(
            {
                "currency": c,
                "weight": float(w.reindex(members).fillna(0.0).sum()),
                "fx_contribution": float(fx_part[members].sum().sum()),
                "local_contribution": float(local_part[members].sum().sum()),
            }
        )
    rows.sort(key=lambda r: -r["weight"])
    return AnalysisOutput(
        "table", {"base_currency": base, "rows": rows, "fx_total": float(fx_part.sum().sum())}
    )


class LiquidityParams(BaseModel):
    liquidate: float = Field(1.0, gt=0, le=1, description="Share of each position to sell")
    adv_share: float = Field(
        0.2, gt=0, le=1, description="Max share of average daily volume per day"
    )


@analysis(
    id="P-021",
    version="1.0.0",
    title="Portfolio liquidity",
    description=(
        "Days to liquidate x % of each position trading at most y % of its 63-session average "
        "daily traded value (crypto volume is already quoted in value); the portfolio needs the "
        "slowest line."
    ),
    applies_to=PORTFOLIO,
    params=LiquidityParams,
    result_kind="table",
    requires=NEEDS,
)
def liquidity(inputs: AnalysisInputs, params: LiquidityParams) -> AnalysisOutput:
    pf = _pf(inputs)
    hv = pf.path.holdings_value
    if hv is None:
        raise InsufficientDataError("No holdings values")
    rows: list[dict[str, Any]] = []
    for s in _assets(pf):
        frame, meta = pf.prices.get(s), pf.meta.get(s) or {}
        if frame is None or "volume" not in frame or frame["volume"].tail(63).isna().all():
            rows.append(
                {"symbol": s, "value": float(hv[s].iloc[-1]), "adv_value": None, "days": None}
            )
            continue
        tail = frame.tail(63)
        traded = (
            tail["volume"]
            if meta.get("asset_class") == "crypto"
            else tail["volume"] * tail["close"]
        )
        rate = pf.fx_last.get(str(meta.get("currency")), 1.0)
        adv = float(traded.mean()) * rate
        value = float(hv[s].iloc[-1])
        days = params.liquidate * abs(value) / (params.adv_share * adv) if adv > 0 else None
        rows.append({"symbol": s, "value": value, "adv_value": adv, "days": days})
    rows.sort(key=lambda r: -(r["days"] or 0))
    slowest = [r["days"] for r in rows if r["days"] is not None]
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "days_total": max(slowest) if slowest else None,
            "within_1_day": sum(
                r["value"] for r in rows if r["days"] is not None and r["days"] <= 1
            )
            / float(hv.iloc[-1].sum()),
        },
    )


# ── P-022 / P-023 scenarios ──────────────────────────────────────────────────


@analysis(
    id="P-022",
    version="1.0.0",
    title="Historical scenario replay",
    description=(
        "Current weights applied to each holding's cumulative total return (listing currency) "
        "inside the R-016 crisis windows: portfolio return, per-holding contributions and the "
        "covered weight (holdings without history in a window are left out)."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def scenario_replay(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    w = _current(pf)
    rows: list[dict[str, Any]] = []
    for name, start, end in STRESS_WINDOWS:
        contrib, covered = {}, float(w.get(CASH, 0.0))
        for s in _assets(pf):
            frame = pf.prices.get(s)
            if (
                frame is None
                or frame.empty
                or frame.index[0] > pd.Timestamp(start)
                or frame.index[-1] < pd.Timestamp(end)
            ):
                continue
            px = frame["adj_close"]
            r = float(px.asof(pd.Timestamp(end)) / px.asof(pd.Timestamp(start)) - 1)
            contrib[s] = float(w.get(s, 0.0)) * r
            covered += float(w.get(s, 0.0))
        if covered - float(w.get(CASH, 0.0)) <= 0:
            continue
        top = sorted(contrib.items(), key=lambda kv: kv[1])[:5]
        rows.append(
            {
                "scenario": name,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "portfolio_return": sum(contrib.values()),
                "covered_weight": covered,
                "worst": [{"symbol": s, "contribution": v} for s, v in top],
            }
        )
    if not rows:
        raise InsufficientDataError("No holding has history covering a crisis window")
    return AnalysisOutput(
        "table",
        {"scenarios": rows},
        ("Returns in listing currency; FX moves in crises are not replayed.",),
    )


FACTOR_SHOCKS = {"equity": -0.20, "rates_10y": 1.00, "usd": 0.10, "oil": 0.30, "credit": 2.00}
FACTOR_UNITS = {
    "equity": "return",
    "rates_10y": "pp",
    "usd": "return",
    "oil": "return",
    "credit": "pp",
}


class FactorParams(BaseModel):
    equity: float = Field(-0.20, description="Equity market shock (return)")
    rates_10y: float = Field(1.00, description="10-year yield shock (percentage points)")
    usd: float = Field(0.10, description="Dollar index shock (return)")
    oil: float = Field(0.30, description="Oil shock (return)")
    credit: float = Field(2.00, description="HY OAS shock (percentage points)")
    weeks: int = Field(156, ge=52, le=520)


@analysis(
    id="P-023",
    version="1.0.0",
    title="Factor-shock scenarios",
    description=(
        "OLS of weekly portfolio NAV returns on weekly factor moves (S&P 500 return, Δ10y yield, "
        "dollar index return, oil return, ΔHY OAS) over the last n weeks; Δvalue = Σ β_k shock_k "
        "per scenario and combined, with t-stats and R²."
    ),
    applies_to=PORTFOLIO,
    params=FactorParams,
    result_kind="table",
    requires=NEEDS | {"factors"},
)
def factor_shocks(inputs: AnalysisInputs, params: FactorParams) -> AnalysisOutput:
    pf = _pf(inputs)
    factors: dict[str, pd.Series] = inputs.data.get("factors") or {}
    missing = sorted(set(FACTOR_SHOCKS) - set(factors))
    if not factors:
        raise InsufficientDataError(
            "Ingest factor proxies (SPY, ^TNX, DX-Y.NYB, CL=F; FRED BAMLH0A0HYM2) for factor shocks"
        )
    weekly = pf.path.nav.resample("W-FRI").last().pct_change()
    cols = {}
    for name, series in factors.items():
        level = series.resample("W-FRI").last()
        cols[name] = level.diff() if FACTOR_UNITS[name] == "pp" else level.pct_change()
    data = pd.concat([weekly.rename("y"), pd.DataFrame(cols)], axis=1).dropna().tail(params.weeks)
    names = list(cols)
    if len(data) < max(26, 3 * len(names)):
        raise InsufficientDataError(f"Needs ≥ 26 common weeks with the factors, got {len(data)}")
    x = np.column_stack([np.ones(len(data)), data[names].to_numpy()])
    y = data["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    se = np.sqrt(np.diag(float(resid @ resid) / dof * np.linalg.pinv(x.T @ x)))
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    shocks = {k: getattr(params, k) for k in names}
    rows = [
        {
            "factor": k,
            "beta": float(b),
            "t_stat": float(b / s) if s else None,
            "shock": shocks[k],
            "impact": float(b * shocks[k]),
        }
        for k, b, s in zip(names, beta[1:], se[1:], strict=True)
    ]
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "combined_impact": float(sum(r["impact"] for r in rows)),
            "r_squared": r2,
            "weeks": len(data),
            "missing_factors": missing,
        },
        ("Linear betas: second-order and regime effects are not captured.",),
    )


# ── P-024 income ─────────────────────────────────────────────────────────────


@analysis(
    id="P-024",
    version="1.0.0",
    title="Income",
    description=(
        "Dividends and interest received (ledger, or implied quantities × stored dividends in "
        "weights mode) by month and by holding; forward 12-month income = quantity × trailing "
        "12-month dividends per share; current yield and yield on cost."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
)
def income(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    path = pf.path
    hv = path.holdings_value
    end = path.value.index[-1]
    received: list[dict[str, Any]] = list(path.income)
    forward = 0.0
    by_holding: dict[str, float] = {}
    for s in _assets(pf):
        acts = pf.actions.get(s)
        divs = (
            acts.loc[acts["action"] == "dividend"] if acts is not None and not acts.empty else None
        )
        rate = pf.fx_last.get(str((pf.meta.get(s) or {}).get("currency")), 1.0)
        frame = pf.prices.get(s)
        if path.quantities is not None and s in path.quantities:
            qty = path.quantities[s]
        elif hv is not None and frame is not None:
            close = frame["close"].reindex(hv.index).ffill()
            qty = hv[s] / (close * rate)
        else:
            continue
        if divs is None or divs.empty:
            continue
        dates = pd.to_datetime(divs["ex_date"])
        if path.quantities is None:  # weights mode: implied receipts
            for d, amount in zip(dates, divs["amount"], strict=True):
                if path.value.index[0] < d <= end:
                    held = float(qty.asof(d - pd.Timedelta(days=1)))
                    received.append(
                        {
                            "date": d.date().isoformat(),
                            "type": "DIVIDEND",
                            "symbol": s,
                            "amount": held * float(amount) * rate,
                            "currency": None,
                        }
                    )
        ttm = float(divs.loc[dates > end - pd.Timedelta(days=365), "amount"].sum())
        line = float(qty.iloc[-1]) * ttm * rate
        forward += line
        by_holding[s] = line
    frame_in = pd.DataFrame(received)
    monthly = []
    if not frame_in.empty:
        frame_in["month"] = pd.to_datetime(frame_in["date"]).dt.to_period("M").astype(str)
        monthly = [
            {"month": m, "income": float(v)}
            for m, v in frame_in.groupby("month")["amount"].sum().tail(36).items()
        ]
    cost = sum(lot["quantity"] * lot["unit_cost"] for lot in path.lots) if path.lots else None
    value = float(path.value.iloc[-1])
    last_year = (
        frame_in.loc[
            pd.to_datetime(frame_in["date"]) > end - pd.Timedelta(days=365), "amount"
        ].sum()
        if not frame_in.empty
        else 0.0
    )
    return AnalysisOutput(
        "table",
        {
            "monthly": monthly,
            "trailing_12m": float(last_year),
            "forward_12m": forward,
            "current_yield": forward / value if value else None,
            "yield_on_cost": forward / cost if cost else None,
            "by_holding": [
                {"symbol": s, "forward_income": v}
                for s, v in sorted(by_holding.items(), key=lambda kv: -kv[1])
            ],
        },
    )


# ── C-001…C-007 interactions ─────────────────────────────────────────────────


class CorrParams(BaseModel):
    method: Literal["pearson", "spearman", "kendall"] = "pearson"
    freq: Literal["D", "W", "M"] = "D"


def _resampled(r: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return r
    rule = {"W": "W-FRI", "M": "ME"}[freq]
    return (1 + r).resample(rule).prod() - 1


@analysis(
    id="C-001",
    version="1.0.0",
    title="Correlation matrix",
    description=(
        "Pearson, Spearman or Kendall correlation of holdings' returns at D/W/M frequency over "
        "the window, ordered by average-linkage clustering on d = √(½(1 − rho)), with the "
        "dendrogram linkage."
    ),
    applies_to=PORTFOLIO,
    params=CorrParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=30,
)
def correlation_matrix(inputs: AnalysisInputs, params: CorrParams) -> AnalysisOutput:
    r = _resampled(_risk_returns(inputs, _pf(inputs)), params.freq)
    if r.shape[1] < 2:
        raise InsufficientDataError("Needs ≥ 2 holdings")
    corr = r.corr(method=params.method).fillna(0.0)
    dist = np.sqrt(np.clip(0.5 * (1 - corr.to_numpy()), 0, None))
    np.fill_diagonal(dist, 0.0)
    link = hierarchy.linkage(squareform(dist, checks=False), method="average")
    order = [corr.columns[i] for i in hierarchy.leaves_list(link)]
    ordered = corr.loc[order, order]
    return AnalysisOutput(
        "table",
        {
            "labels": order,
            "matrix": ordered.to_numpy().tolist(),
            "method": params.method,
            "freq": params.freq,
            "linkage": link.tolist(),
            "observations": len(r),
            "average_correlation": float(corr.to_numpy()[np.triu_indices(len(corr), 1)].mean()),
        },
    )


class PairParams(BaseModel):
    a: str | None = Field(None, max_length=20)
    b: str | None = Field(None, max_length=20)


@analysis(
    id="C-002",
    version="1.0.0",
    title="Rolling correlation",
    description=(
        "Rolling 63/126/252-session correlation of a pair of holdings (default: the two largest)."
    ),
    applies_to=PORTFOLIO,
    params=PairParams,
    result_kind="series",
    requires=NEEDS,
    min_obs=63,
)
def rolling_pair(inputs: AnalysisInputs, params: PairParams) -> AnalysisOutput:
    pf = _pf(inputs)
    r = pf.path.returns[_assets(pf)].iloc[1:]
    ranked = list(_current(pf).drop(CASH, errors="ignore").sort_values(ascending=False).index)
    a = (params.a or (ranked[0] if ranked else "")).upper()
    b = (params.b or (ranked[1] if len(ranked) > 1 else "")).upper()
    if a not in r or b not in r or a == b:
        raise InsufficientDataError(f"Pick two different holdings among {sorted(r.columns)}")
    lines = {f"corr_{n}": r[a].rolling(n).corr(r[b]) for n in (63, 126, 252)}
    out = _window(inputs, pd.DataFrame(lines))
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(out.index),
            "a": a,
            "b": b,
            "lines": {k: [None if not np.isfinite(v) else float(v) for v in out[k]] for k in out},
        },
    )


def _avg_corr(r: pd.DataFrame) -> float:
    c = r.corr().to_numpy()
    return float(np.nanmean(c[np.triu_indices(len(c), 1)]))


@analysis(
    id="C-003",
    version="1.0.0",
    title="Stress correlation",
    description=(
        "Average pairwise correlation of holdings on the worst 10 % days of the benchmark (or of "
        "the portfolio when no benchmark is stored) vs all days of the window."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS | {"other_prices"},
    min_obs=60,
)
def stress_correlation(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    r = _risk_returns(inputs, pf, 60)
    if r.shape[1] < 2:
        raise InsufficientDataError("Needs ≥ 2 holdings")
    other = inputs.data.get("other_prices")
    if isinstance(other, pd.DataFrame) and not other.empty:
        driver = other["adj_close"].pct_change().reindex(r.index)
        source = str(inputs.conventions.get("other_symbol"))
    else:
        driver = pf.path.nav.pct_change().reindex(r.index)
        source = "portfolio"
    stress = driver <= driver.quantile(0.10)
    calm = r.loc[stress]
    return AnalysisOutput(
        "table",
        {
            "all_days": _avg_corr(r),
            "stress_days": _avg_corr(calm),
            "stress_count": int(stress.sum()),
            "driver": source,
            "labels": list(r.columns),
            "stress_matrix": calm.corr().fillna(0.0).to_numpy().tolist(),
        },
    )


@analysis(
    id="C-004",
    version="1.0.0",
    title="Risk–return map",
    description=(
        "Annualised volatility vs CAGR of each holding and of the portfolio over the window, "
        "bubble = weight, with iso-Sharpe lines through the risk-free rate."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=30,
)
def risk_return_map(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    pf = _pf(inputs)
    r = _risk_returns(inputs, pf)
    factor = _factor(inputs)
    years = len(r) / factor
    w = _current(pf)

    def point(x: pd.Series) -> tuple[float, float]:
        return float(x.std() * math.sqrt(factor)), float((1 + x).prod() ** (1 / years) - 1)

    points = []
    for s in r.columns:
        vol, cagr = point(r[s])
        points.append(
            {"symbol": s, "volatility": vol, "cagr": cagr, "weight": float(w.get(s, 0.0))}
        )
    nav = _window(inputs, pf.path.nav).pct_change().dropna()
    vol_p, cagr_p = point(nav)
    rf = float(inputs.conventions.get("rf") or 0.0)
    return AnalysisOutput(
        "table",
        {
            "points": points,
            "portfolio": {"volatility": vol_p, "cagr": cagr_p},
            "rf": rf,
            "portfolio_sharpe": (cagr_p - rf) / vol_p if vol_p else None,
            "iso_sharpe": [0.5, 1.0],
        },
    )


@analysis(
    id="C-005",
    version="1.0.0",
    title="Correlation stability",
    description=(
        "Correlation matrix of the second half of the window minus the first half, and the pairs "
        "that changed most."
    ),
    applies_to=PORTFOLIO,
    params=NoParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=60,
)
def correlation_stability(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    r = _risk_returns(inputs, _pf(inputs), 60)
    if r.shape[1] < 2:
        raise InsufficientDataError("Needs ≥ 2 holdings")
    half = len(r) // 2
    first, second = r.iloc[:half].corr(), r.iloc[half:].corr()
    diff = (second - first).fillna(0.0)
    pairs = [
        {
            "a": a,
            "b": b,
            "first": float(first.at[a, b]),
            "second": float(second.at[a, b]),
            "change": float(diff.at[a, b]),
        }
        for a, b in itertools.combinations(r.columns, 2)
    ]
    pairs.sort(key=lambda p: -abs(p["change"]))
    return AnalysisOutput(
        "table",
        {
            "labels": list(r.columns),
            "difference": diff.to_numpy().tolist(),
            "changed": pairs[:10],
            "split": _iso(r.index[[half]])[0],
            "mean_abs_change": float(np.abs(diff.to_numpy()[np.triu_indices(len(diff), 1)]).mean()),
        },
    )


class TailParams(BaseModel):
    alpha: float = Field(0.05, gt=0.005, le=0.2)


@analysis(
    id="C-007",
    version="1.0.0",
    title="Tail dependence",
    description=(
        "Empirical lower-tail dependence λ̂ = P(r_j ≤ q_j(α) | r_i ≤ q_i(α)) for each pair over "
        "the window (independence gives α), with the most co-crashing pairs."
    ),
    applies_to=PORTFOLIO,
    params=TailParams,
    result_kind="table",
    requires=NEEDS,
    min_obs=100,
)
def tail_dependence(inputs: AnalysisInputs, params: TailParams) -> AnalysisOutput:
    r = _risk_returns(inputs, _pf(inputs), 100)
    if r.shape[1] < 2:
        raise InsufficientDataError("Needs ≥ 2 holdings")
    low = r.le(r.quantile(params.alpha))
    labels = list(r.columns)
    m = np.eye(len(labels))
    pairs = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        both = float((low.iloc[:, i] & low.iloc[:, j]).sum())
        lam = both / max(float(low.iloc[:, i].sum()), 1.0)
        m[i, j] = m[j, i] = lam
        pairs.append({"a": labels[i], "b": labels[j], "lambda": lam})
    pairs.sort(key=lambda p: -p["lambda"])
    return AnalysisOutput(
        "table",
        {
            "labels": labels,
            "matrix": m.tolist(),
            "alpha": params.alpha,
            "independence": params.alpha,
            "top": pairs[:10],
        },
    )
