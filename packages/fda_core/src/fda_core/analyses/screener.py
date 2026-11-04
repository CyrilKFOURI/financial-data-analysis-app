"""Screener & universe analytics (ANALYTICS_CATALOG SC-002…SC-008, MK-012; module M20).

Subject `universe` (id = universe id). Inputs:
- `snapshot`: one row per member (symbol, name, sector + metric columns, `METRICS`), as of the
  snapshot date in `conventions["as_of"]`;
- `member_prices`: {symbol: adjusted close} for breadth (MK-012).
Members lacking a metric are excluded from conditions on it and counted (coverage), never
imputed. Screens are descriptive filters, never recommendations.
"""

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

UNIVERSE: set[Any] = {"universe"}

# key → (label, unit, direction: +1 higher is "more of the factor", −1 lower, 0 neutral, group)
METRICS: dict[str, tuple[str, str, int, str]] = {
    "last_price": ("Last price", "number", 0, "price"),
    "return_1m": ("Return 1M", "percent", 1, "price"),
    "return_3m": ("Return 3M", "percent", 1, "price"),
    "return_6m": ("Return 6M", "percent", 1, "price"),
    "return_12m": ("Return 12M", "percent", 1, "price"),
    "momentum_12_1": ("Momentum 12-1", "percent", 1, "price"),
    "return_ytd": ("Return YTD", "percent", 1, "price"),
    "volatility_1y": ("Volatility 1Y", "percent", -1, "price"),
    "max_drawdown_1y": ("Max drawdown 1Y", "percent", 1, "price"),
    "drawdown_from_high": ("Drawdown from high", "percent", 1, "price"),
    "beta_1y": ("Beta 1Y", "number", 0, "price"),
    "position_52w": ("52-week position", "percent", 0, "price"),
    "above_dma50": ("Above 50-DMA", "flag", 0, "price"),
    "above_dma200": ("Above 200-DMA", "flag", 0, "price"),
    "rsi_14": ("RSI 14", "number", 0, "price"),
    "dollar_volume_3m": ("Avg dollar volume 3M", "number", 0, "price"),
    "market_cap": ("Market cap", "number", 0, "valuation"),
    "enterprise_value": ("Enterprise value", "number", 0, "valuation"),
    "pe": ("P/E", "number", -1, "valuation"),
    "ev_ebitda": ("EV/EBITDA", "number", -1, "valuation"),
    "ev_ebit": ("EV/EBIT", "number", -1, "valuation"),
    "ev_sales": ("EV/Sales", "number", -1, "valuation"),
    "p_b": ("P/B", "number", -1, "valuation"),
    "earnings_yield": ("Earnings yield", "percent", 1, "valuation"),
    "fcf_yield": ("FCF yield", "percent", 1, "valuation"),
    "dividend_yield": ("Dividend yield", "percent", 1, "valuation"),
    "gross_margin": ("Gross margin", "percent", 1, "quality"),
    "ebit_margin": ("EBIT margin", "percent", 1, "quality"),
    "net_margin": ("Net margin", "percent", 1, "quality"),
    "fcf_margin": ("FCF margin", "percent", 1, "quality"),
    "roe": ("ROE", "percent", 1, "quality"),
    "roic": ("ROIC", "percent", 1, "quality"),
    "nd_ebitda": ("Net debt / EBITDA", "number", -1, "quality"),
    "debt_to_equity": ("Debt / equity", "number", -1, "quality"),
    "net_cash": ("Net cash", "flag", 0, "quality"),
    "fcf_to_ni": ("FCF / NI", "number", 1, "quality"),
    "piotroski": ("Piotroski F-score", "number", 1, "quality"),
    "health_score": ("Health score", "number", 1, "quality"),
    "revenue_growth": ("Revenue growth YoY", "percent", 1, "growth"),
    "eps_growth": ("EPS growth YoY", "percent", 1, "growth"),
    "revenue_cagr_3y": ("Revenue CAGR 3Y", "percent", 1, "growth"),
    "dividend_years": ("Years of dividend growth", "number", 1, "growth"),
}

# Factor → [(metric, sign)]: z-scores are signed so that higher = more of the factor.
FACTORS: dict[str, list[tuple[str, int]]] = {
    "value": [("earnings_yield", 1), ("fcf_yield", 1), ("ev_ebit", -1), ("p_b", -1)],
    "quality": [("roic", 1), ("gross_margin", 1), ("fcf_to_ni", 1), ("nd_ebitda", -1)],
    "momentum": [("momentum_12_1", 1)],
    "low_vol": [("volatility_1y", -1)],
    "growth": [("revenue_growth", 1), ("eps_growth", 1), ("revenue_cagr_3y", 1)],
    "size": [("market_cap", -1)],
    "yield": [("dividend_yield", 1)],
}

PRESETS: dict[str, dict[str, Any]] = {
    "quality_compounders": {"label": "Quality compounders", "filters": {"all": [
        {"metric": "roic", "op": ">", "value": 0.15},
        {"metric": "revenue_cagr_3y", "op": ">", "value": 0.07},
        {"metric": "nd_ebitda", "op": "<", "value": 2},
        {"metric": "fcf_to_ni", "op": ">", "value": 0.8}]}},
    "net_cash_value": {"label": "Net-cash value", "filters": {"all": [
        {"metric": "net_cash", "op": "=", "value": 1},
        {"metric": "fcf_yield", "op": ">", "value": 0.05},
        {"metric": "ev_ebit", "op": "<", "value": 30, "basis": "sector_percentile"}]}},
    "dividend_growers": {"label": "Dividend growers (≥ 10 y)", "filters": {"all": [
        {"metric": "dividend_years", "op": ">=", "value": 10}]}},
    "high_accruals": {"label": "Cash-quality warning list", "filters": {"all": [
        {"metric": "fcf_to_ni", "op": "<", "value": 0.5},
        {"metric": "net_margin", "op": ">", "value": 0}]}},
    "deep_drawdown": {"label": "Deep drawdown list", "filters": {"all": [
        {"metric": "drawdown_from_high", "op": "<", "value": -0.4}]}},
    "low_vol_quality": {"label": "Low-vol high-quality", "filters": {"all": [
        {"metric": "volatility_1y", "op": "<", "value": 30, "basis": "percentile"},
        {"metric": "roic", "op": ">", "value": 60, "basis": "percentile"}]}},
}  # fmt: skip


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── SC-002 price metrics (per member, pure) ──────────────────────────────────


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    delta = close.diff().dropna()
    if len(delta) < period * 2:
        return None
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    total = gain + loss
    return 50.0 if total == 0 else float(100 * gain / total)


def price_metrics(
    frame: pd.DataFrame, bench: pd.Series | None, annualisation: int = 252
) -> dict[str, Any]:
    """Snapshot metrics from adjusted closes (and volume): calendar-anchored returns, 12-1
    momentum (t−252 → t−21 sessions), 1-year volatility, drawdowns, beta vs the benchmark,
    52-week position, moving-average states, RSI(14), 63-session average dollar volume."""
    close = frame["adj_close"].astype(float).dropna()
    if len(close) < 2:
        return {}
    last, last_date = float(close.iloc[-1]), close.index[-1]

    def back(days: int) -> float | None:
        before = close.loc[: last_date - pd.Timedelta(days=days)]
        return (
            float(before.iloc[-1])
            if len(before) and close.index[0] <= last_date - pd.Timedelta(days=days - 5)
            else None
        )

    out: dict[str, Any] = {
        "last_price": float(frame["close"].iloc[-1]) if "close" in frame else last
    }
    for key, days in (
        ("return_1m", 31),
        ("return_3m", 92),
        ("return_6m", 183),
        ("return_12m", 365),
    ):
        base = back(days)
        out[key] = last / base - 1 if base else None
    prior_year = close.loc[: pd.Timestamp(last_date.year, 1, 1) - pd.Timedelta(days=1)]
    out["return_ytd"] = last / float(prior_year.iloc[-1]) - 1 if len(prior_year) else None
    out["momentum_12_1"] = (
        float(close.iloc[-22] / close.iloc[-253] - 1) if len(close) > 253 else None
    )
    year = close.loc[close.index > last_date - pd.Timedelta(days=365)]
    r = np.log(year).diff().dropna()
    out["volatility_1y"] = float(r.std() * math.sqrt(annualisation)) if len(r) > 120 else None
    out["max_drawdown_1y"] = float((year / year.cummax() - 1).min()) if len(year) > 20 else None
    out["drawdown_from_high"] = last / float(close.max()) - 1
    lo, hi = float(year.min()), float(year.max())
    out["position_52w"] = (last - lo) / (hi - lo) if hi > lo else None
    out["above_dma50"] = float(last > close.iloc[-50:].mean()) if len(close) >= 50 else None
    out["above_dma200"] = float(last > close.iloc[-200:].mean()) if len(close) >= 200 else None
    out["rsi_14"] = _rsi(close)
    if "volume" in frame and frame["volume"].notna().any():
        dv = (frame["close"] * frame["volume"]).iloc[-63:]
        out["dollar_volume_3m"] = _f(dv.mean())
    if bench is not None and len(r) > 120:
        both = pd.concat([r, np.log(bench.astype(float)).diff()], axis=1, join="inner").dropna()
        var = both.iloc[:, 1].var()
        out["beta_1y"] = float(both.cov().iloc[0, 1] / var) if len(both) > 120 and var > 0 else None
    return {k: _f(v) for k, v in out.items()}


# ── SC-003 screen ────────────────────────────────────────────────────────────

Basis = Literal["absolute", "percentile", "sector_percentile"]
OPS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "=": lambda a, b: np.isclose(a, b),
    "!=": lambda a, b: ~np.isclose(a, b),
}


def percentile_rank(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """0–100 rank (average ties) within the universe or within each group; NaN stays NaN."""
    if groups is None:
        return values.rank(pct=True) * 100
    return values.groupby(groups.fillna("Unclassified")).rank(pct=True) * 100


def _condition(frame: pd.DataFrame, cond: dict[str, Any], coverage: dict[str, int]) -> pd.Series:
    metric = str(cond.get("metric"))
    if metric not in frame.columns:
        raise InsufficientDataError(f"Unknown metric {metric!r} (not in the snapshot)")
    op = str(cond.get("op", ">"))
    if op not in OPS and op != "between":
        raise InsufficientDataError(f"Unknown operator {op!r}")
    basis = cond.get("basis", "absolute")
    values = pd.to_numeric(frame[metric], errors="coerce")
    if basis == "percentile":
        values = percentile_rank(values)
    elif basis == "sector_percentile":
        values = percentile_rank(values, frame.get("sector"))
    coverage[metric] = int(values.notna().sum())
    known = values.notna()
    if op == "between":
        lo, hi = float(cond["value"]), float(cond["value2"])
        result = (values >= lo) & (values <= hi)
    else:
        result = pd.Series(
            OPS[op](values.to_numpy(dtype=float), float(cond["value"])),
            index=frame.index,
        )
    return result & known


def evaluate(frame: pd.DataFrame, tree: dict[str, Any], coverage: dict[str, int]) -> pd.Series:
    """Filter tree: {"all": [...]} (AND) or {"any": [...]} (OR); leaves are conditions
    {metric, op (> >= < <= = != between), value, value2?, basis (absolute | percentile |
    sector_percentile)}. An empty tree keeps every member."""
    if "metric" in tree:
        return _condition(frame, tree, coverage)
    mode = "all" if "all" in tree else "any"
    children = tree.get(mode) or []
    if not children:
        return pd.Series(True, index=frame.index)
    masks = [evaluate(frame, child, coverage) for child in children]
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask if mode == "all" else combined | mask
    return combined


class ScreenParams(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)
    preset: str | None = Field(None, description="Key of a preset screen (SC-008)")
    columns: list[str] = Field(default_factory=lambda: ["market_cap", "pe", "roic", "return_12m"])
    sort: str | None = None
    descending: bool = True
    limit: int = Field(200, ge=1, le=2000)


def run_screen(frame: pd.DataFrame, params: ScreenParams) -> dict[str, Any]:
    tree = params.filters
    if params.preset:
        if params.preset not in PRESETS:
            raise InsufficientDataError(f"Unknown preset {params.preset!r}")
        tree = PRESETS[params.preset]["filters"]
    coverage: dict[str, int] = {}
    mask = evaluate(frame, tree, coverage)
    hits = frame.loc[mask]
    sort = (
        params.sort
        if params.sort in frame.columns
        else (params.columns[0] if params.columns and params.columns[0] in frame.columns else None)
    )
    if sort:
        hits = hits.sort_values(sort, ascending=not params.descending, na_position="last")
    columns = [c for c in params.columns if c in frame.columns]
    rows = [
        {
            "symbol": r["symbol"],
            "name": r.get("name"),
            "sector": r.get("sector"),
            **{c: _f(r.get(c)) for c in columns},
        }
        for r in hits.head(params.limit).to_dict("records")
    ]
    mix = (
        hits["sector"].fillna("Unclassified").value_counts()
        if "sector" in hits
        else pd.Series(dtype=int)
    )
    return {
        "count": int(mask.sum()),
        "universe_size": len(frame),
        "coverage": {m: {"with_data": n, "excluded": len(frame) - n} for m, n in coverage.items()},
        "columns": columns,
        "rows": rows,
        "sector_mix": [{"sector": s, "count": int(n)} for s, n in mix.items()],
        "symbols": sorted(hits["symbol"].astype(str)),
    }


def screen_diff(previous: list[str], current: list[str]) -> dict[str, list[str]]:
    """SC-007: entrants and leavers between two runs of a saved screen."""
    return {
        "entrants": sorted(set(current) - set(previous)),
        "leavers": sorted(set(previous) - set(current)),
    }


def _snapshot(inputs: AnalysisInputs) -> pd.DataFrame:
    frame = inputs.data.get("snapshot")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise InsufficientDataError(
            "No snapshot for this universe: build one (POST /v1/universes/{id}/snapshot)"
        )
    return frame


@analysis(
    id="SC-003",
    version="1.0.0",
    title="Screen",
    description=(
        "AND/OR filter tree on snapshot metrics: absolute thresholds, percentile within the "
        "universe or within the sector (0–100, average ties). Members lacking a metric fail "
        "conditions on it and are counted per metric (coverage). Sorted results, sector mix. "
        "Presets (SC-008) are descriptive filters, not recommendations."
    ),
    applies_to=UNIVERSE,
    params=ScreenParams,
    result_kind="table",
    requires={"snapshot"},
)
def screen(inputs: AnalysisInputs, params: ScreenParams) -> AnalysisOutput:
    out = run_screen(_snapshot(inputs), params)
    return AnalysisOutput("table", {"as_of": inputs.conventions.get("as_of"), **out})


# ── SC-004 factor ranks ──────────────────────────────────────────────────────


def winsorise(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    known = values.dropna()
    if len(known) < 5:
        return values
    return values.clip(known.quantile(lower), known.quantile(upper))


def zscore(values: pd.Series, groups: pd.Series | None = None, min_group: int = 3) -> pd.Series:
    """Winsorised (1/99) z-score; sector-neutral when groups are given (groups smaller than
    `min_group` fall back to the universe mean and standard deviation)."""
    w = winsorise(values.astype(float))
    universe = (w - w.mean()) / w.std(ddof=0) if w.std(ddof=0) > 0 else w * 0
    if groups is None:
        return universe
    g = groups.fillna("Unclassified")
    sizes = w.groupby(g).transform("count")
    mean = w.groupby(g).transform("mean")
    std = w.groupby(g).transform(lambda s: s.std(ddof=0))
    neutral = (w - mean) / std.replace(0, np.nan)
    return neutral.where(sizes >= min_group, universe)


class RankParams(BaseModel):
    sector_neutral: bool = False
    weights: dict[str, float] = Field(default_factory=lambda: {f: 1.0 for f in FACTORS})
    limit: int = Field(500, ge=1, le=2000)


def factor_scores(frame: pd.DataFrame, params: RankParams) -> pd.DataFrame:
    groups = frame["sector"] if params.sector_neutral and "sector" in frame else None
    scores = pd.DataFrame(index=frame.index)
    for factor, metrics in FACTORS.items():
        parts = [
            sign * zscore(pd.to_numeric(frame[m], errors="coerce"), groups)
            for m, sign in metrics
            if m in frame
        ]
        if parts:
            scores[factor] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    weights = {f: w for f, w in params.weights.items() if f in scores and w != 0}
    if weights:
        weighted = sum(scores[f] * w for f, w in weights.items())
        present = sum(scores[f].notna() * abs(w) for f, w in weights.items())
        scores["composite"] = weighted / present.replace(0, np.nan)
    return scores


@analysis(
    id="SC-004",
    version="1.0.0",
    title="Factor ranks",
    description=(
        "Per member: value (earnings & FCF yield, EV/EBIT, P/B), quality (ROIC, gross margin, "
        "FCF/NI, ND/EBITDA), momentum (12-1), low volatility, growth (revenue/EPS YoY, revenue "
        "CAGR 3y), size (smaller = higher), yield. Metrics winsorised 1/99, z-scored (optionally "
        "within sector), signed, averaged per factor; composite = weighted mean; percentiles; "
        "factor correlation matrix. Revisions need estimate history (not stored)."
    ),
    applies_to=UNIVERSE,
    params=RankParams,
    result_kind="table",
    requires={"snapshot"},
)
def factor_ranks(inputs: AnalysisInputs, params: RankParams) -> AnalysisOutput:
    frame = _snapshot(inputs)
    scores = factor_scores(frame, params)
    pct = scores.rank(pct=True) * 100
    order = (
        scores["composite"].sort_values(ascending=False).index
        if "composite" in scores
        else scores.index
    )
    rows = [
        {
            "symbol": frame.at[i, "symbol"],
            "name": frame.at[i, "name"] if "name" in frame else None,
            "sector": frame.at[i, "sector"] if "sector" in frame else None,
            **{f"{c}_z": _f(scores.at[i, c]) for c in scores.columns},
            **{f"{c}_pct": _f(pct.at[i, c]) for c in scores.columns},
        }
        for i in list(order)[: params.limit]
    ]
    corr = scores.drop(columns="composite", errors="ignore").corr()
    return AnalysisOutput(
        "table",
        {
            "as_of": inputs.conventions.get("as_of"),
            "sector_neutral": params.sector_neutral,
            "rows": rows,
            "coverage": {c: int(scores[c].notna().sum()) for c in scores.columns},
            "grid": {
                "rows": list(corr.index),
                "columns": list(corr.columns),
                "values": [[_f(v) for v in row] for row in corr.to_numpy()],
                "unit": "number",
                "best": [],
            },
        },
    )


# ── SC-005 heatmap / SC-006 distributions ────────────────────────────────────


class MetricParams(BaseModel):
    metric: str = "return_1m"
    size: str = "market_cap"
    highlight: str | None = Field(None, max_length=20)
    bins: int = Field(30, ge=5, le=100)


@analysis(
    id="SC-005",
    version="1.0.0",
    title="Universe heatmap",
    description=(
        "Treemap grouped by sector, tiles sized by market cap (equal size when unknown) and "
        "coloured by the chosen metric; colour scale centred on 0 for returns, else on the "
        "universe median."
    ),
    applies_to=UNIVERSE,
    params=MetricParams,
    result_kind="table",
    requires={"snapshot"},
)
def heatmap(inputs: AnalysisInputs, params: MetricParams) -> AnalysisOutput:
    frame = _snapshot(inputs)
    if params.metric not in frame:
        raise InsufficientDataError(f"Unknown metric {params.metric!r}")
    metric = pd.to_numeric(frame[params.metric], errors="coerce")
    size = (
        pd.to_numeric(frame[params.size], errors="coerce")
        if params.size in frame
        else pd.Series(np.nan, index=frame.index)
    )
    groups = []
    for sector, idx in frame.groupby(
        frame.get("sector", pd.Series("All", index=frame.index)).fillna("Unclassified")
    ).groups.items():
        children = [
            {
                "name": frame.at[i, "symbol"],
                "value": _f(size.at[i]) or 1.0,
                "metric": _f(metric.at[i]),
            }
            for i in idx
        ]
        groups.append(
            {"name": sector, "value": sum(c["value"] for c in children), "children": children}
        )
    unit = METRICS.get(params.metric, ("", "number", 0, ""))[1]
    centre = (
        0.0
        if params.metric.startswith("return") or "momentum" in params.metric
        else _f(metric.median())
    )
    return AnalysisOutput(
        "table",
        {
            "metric": params.metric,
            "unit": unit,
            "centre": centre,
            "size": params.size,
            "tree": sorted(groups, key=lambda g: -g["value"]),
            "range": [_f(metric.quantile(0.02)), _f(metric.quantile(0.98))],
        },
    )


@analysis(
    id="SC-006",
    version="1.0.0",
    title="Metric distribution",
    description=(
        "Histogram of a metric across the universe (bins between the 1st and 99th percentiles, "
        "tails counted at the edges) and per-sector quantiles (p10, p25, median, p75, p90); the "
        "highlighted member's value and percentile."
    ),
    applies_to=UNIVERSE,
    params=MetricParams,
    result_kind="table",
    requires={"snapshot"},
)
def distribution(inputs: AnalysisInputs, params: MetricParams) -> AnalysisOutput:
    frame = _snapshot(inputs)
    if params.metric not in frame:
        raise InsufficientDataError(f"Unknown metric {params.metric!r}")
    values = pd.to_numeric(frame[params.metric], errors="coerce")
    known = values.dropna()
    if len(known) < 5:
        raise InsufficientDataError(
            f"{params.metric}: needs ≥ 5 members with data, got {len(known)}"
        )
    lo, hi = float(known.quantile(0.01)), float(known.quantile(0.99))
    counts, edges = np.histogram(
        known.clip(lo, hi), bins=params.bins, range=(lo, hi) if hi > lo else None
    )
    sectors = []
    for sector, s in values.groupby(
        frame.get("sector", pd.Series("All", index=frame.index)).fillna("Unclassified")
    ):
        s = s.dropna()
        if len(s):
            sectors.append(
                {
                    "sector": sector,
                    "n": len(s),
                    **{
                        f"p{int(q * 100)}": float(s.quantile(q))
                        for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                    },
                }
            )
    mark = None
    if params.highlight:
        hit = frame.index[frame["symbol"].astype(str).str.upper() == params.highlight.upper()]
        if len(hit) and pd.notna(values.at[hit[0]]):
            v = float(values.at[hit[0]])
            mark = {
                "symbol": params.highlight.upper(),
                "value": v,
                "percentile": float((known <= v).mean() * 100),
            }
    return AnalysisOutput(
        "table",
        {
            "metric": params.metric,
            "n": len(known),
            "excluded": int(values.isna().sum()),
            "histogram": [
                {"from": float(a), "to": float(b), "count": int(c)}
                for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)
            ],
            "sectors": sorted(sectors, key=lambda r: -r["n"]),
            "highlight": mark,
            "median": float(known.median()),
        },
    )


# ── MK-012 breadth ───────────────────────────────────────────────────────────


def breadth(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily breadth on the members' common calendar: % above 50/200-DMA (of members with
    enough history), advances − declines and its cumulative A/D line, 52-week new highs/lows,
    McClellan oscillator EMA19 − EMA39 of net advances and its summation index."""
    panel = pd.concat(prices, axis=1).sort_index().ffill(limit=5)
    change = panel.diff()
    advances = (change > 0).sum(axis=1)
    declines = (change < 0).sum(axis=1)
    net = advances - declines
    above50 = (panel > panel.rolling(50).mean()).where(panel.rolling(50).count() == 50)
    above200 = (panel > panel.rolling(200).mean()).where(panel.rolling(200).count() == 200)
    high = panel >= panel.rolling(252, min_periods=200).max()
    low = panel <= panel.rolling(252, min_periods=200).min()
    mcclellan = net.ewm(span=19, adjust=False).mean() - net.ewm(span=39, adjust=False).mean()
    out = pd.DataFrame(
        {
            "pct_above_50dma": above50.mean(axis=1) * 100,
            "pct_above_200dma": above200.mean(axis=1) * 100,
            "advances": advances,
            "declines": declines,
            "ad_line": net.cumsum(),
            "new_highs": high.sum(axis=1),
            "new_lows": low.sum(axis=1),
            "mcclellan": mcclellan,
            "summation": mcclellan.cumsum(),
        }
    )
    return out.iloc[1:]


class BreadthParams(BaseModel):
    days: int = Field(504, ge=60, le=2520)


@analysis(
    id="MK-012",
    version="1.0.0",
    title="Breadth",
    description=(
        "Universe breadth: % of members above their 50- and 200-day averages, advance/decline "
        "line, 52-week new highs and lows, McClellan oscillator (EMA19 − EMA39 of net advances) "
        "and summation index, on stored member prices."
    ),
    applies_to=UNIVERSE,
    params=BreadthParams,
    result_kind="series",
    requires={"member_prices"},
)
def universe_breadth(inputs: AnalysisInputs, params: BreadthParams) -> AnalysisOutput:
    prices = inputs.data.get("member_prices")
    if not isinstance(prices, dict) or len(prices) < 5:
        raise InsufficientDataError(
            "Breadth needs stored prices for ≥ 5 members: ingest the universe"
        )
    frame = breadth(prices).iloc[-params.days :]
    last = frame.iloc[-1]
    return AnalysisOutput(
        "series",
        {
            "members_with_prices": len(prices),
            "pct_above_50dma": _f(last["pct_above_50dma"]),
            "pct_above_200dma": _f(last["pct_above_200dma"]),
            "new_highs": int(last["new_highs"]),
            "new_lows": int(last["new_lows"]),
            "mcclellan": _f(last["mcclellan"]),
            "dates": [d.date().isoformat() for d in pd.to_datetime(frame.index)],
            "lines": {
                c: [_f(v) for v in frame[c]]
                for c in ("pct_above_50dma", "pct_above_200dma", "mcclellan")
            },
            "ad_line": [_f(v) for v in frame["ad_line"]],
        },
    )
