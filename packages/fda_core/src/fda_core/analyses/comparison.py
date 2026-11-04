"""Comparison analyses (ANALYTICS_CATALOG CP-001, CP-003; module M16).

Inputs: `members` — one dict per subject (asset, portfolio, benchmark) with its level series
already converted to the comparison currency and its own annualisation factor. Every metric is
computed on the common window (latest first date, earliest last date, within the context range);
each member keeps its own calendar and annualisation unless `harmonise` is set (then 252).
Matrix-shaped outputs use a `grid` payload: rows × columns with a unit and, for KPIs, the best
column per row.
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from fda_core.analyses.price import window_start
from fda_core.analyses.risk import STRESS_WINDOWS, sharpe_ratio, sortino_ratio
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

COMPARISON: set[Any] = {"comparison"}

View = Literal[
    "kpi",
    "overlay",
    "drawdown",
    "rolling_vol",
    "rolling_sharpe",
    "calendar",
    "correlation",
    "stress",
    "distribution",
]


class CompareParams(BaseModel):
    view: View = "kpi"
    harmonise: bool = Field(False, description="Annualise every member with 252")


def _members(inputs: AnalysisInputs) -> list[dict[str, Any]]:
    members = inputs.data.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise InsufficientDataError("A comparison needs 2–8 subjects")
    return members


def common_window(
    inputs: AnalysisInputs, members: list[dict[str, Any]]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = max(m["series"].first_valid_index() for m in members)
    last = min(m["series"].last_valid_index() for m in members)
    if first >= last:
        raise InsufficientDataError("The members have no common history")
    start = pd.Timestamp(
        window_start(str(inputs.conventions.get("range", "1Y")), last.date(), first.date())
    )
    return max(first, start), last


def _clip(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return series.loc[(series.index >= start) & (series.index <= end)].dropna()


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _grid(
    rows: list[str],
    columns: list[str],
    values: list[list[float | None]],
    unit: str | list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {"rows": rows, "columns": columns, "values": values, "unit": unit} | extra


# (group, metric, unit, direction: +1 higher is better, −1 lower is better, 0 no ranking)
KPIS = [
    ("Return", "total_return", "percent", 1),
    ("Return", "cagr", "percent", 1),
    ("Risk", "volatility", "percent", -1),
    ("Risk", "max_drawdown", "percent", 1),
    ("Risk", "downside_deviation", "percent", -1),
    ("Risk-adjusted", "sharpe", "number", 1),
    ("Risk-adjusted", "sortino", "number", 1),
    ("Risk-adjusted", "calmar", "number", 1),
    ("Tail", "var_95", "percent", -1),
    ("Tail", "es_95", "percent", -1),
    ("Tail", "skewness", "number", 1),
    ("Tail", "excess_kurtosis", "number", -1),
    ("Benchmark-relative", "beta", "number", 0),
    ("Benchmark-relative", "correlation", "number", 0),
    ("Benchmark-relative", "tracking_error", "percent", -1),
    ("Benchmark-relative", "information_ratio", "number", 1),
]


def member_kpis(
    level: pd.Series, factor: int, rf: float, bench: pd.Series | None
) -> dict[str, float | None]:
    r = level.pct_change().dropna()
    if len(r) < 20:
        return {}
    x = r.to_numpy()
    days = (level.index[-1] - level.index[0]).days
    total = float(level.iloc[-1] / level.iloc[0] - 1)
    cagr = (1 + total) ** (365.25 / days) - 1 if days > 0 and total > -1 else None
    dd = float((level / level.cummax() - 1).min())
    downside = math.sqrt(float(np.mean(np.minimum(x, 0) ** 2))) * math.sqrt(factor)
    var = float(-np.quantile(x, 0.05))
    out: dict[str, float | None] = {
        "total_return": total,
        "cagr": cagr,
        "volatility": float(x.std(ddof=1) * math.sqrt(factor)),
        "max_drawdown": dd,
        "downside_deviation": downside,
        "sharpe": sharpe_ratio(x, rf, factor),
        "sortino": sortino_ratio(x, 0.0, factor),
        "calmar": cagr / abs(dd) if cagr is not None and dd else None,
        "var_95": var,
        "es_95": float(-x[x <= -var].mean()),
        "skewness": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
    }
    if bench is not None:
        both = pd.concat([r, bench.pct_change()], axis=1, join="inner").dropna()
        if len(both) >= 20:
            a, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
            active = a - b
            te = float(active.std(ddof=1) * math.sqrt(factor))
            out |= {
                "beta": float(np.cov(a, b, ddof=1)[0, 1] / b.var(ddof=1)),
                "correlation": float(np.corrcoef(a, b)[0, 1]),
                "tracking_error": te,
                "information_ratio": float(active.mean() * factor / te) if te else None,
            }
    return out


def _benchmark(members: list[dict[str, Any]]) -> int | None:
    flagged = [i for i, m in enumerate(members) if m.get("is_benchmark")]
    return flagged[0] if flagged else None


def _union(members: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.concat(
        {m["label"]: _clip(m["series"], start, end) for m in members}, axis=1
    ).sort_index()
    return frame.ffill()


@analysis(
    id="CP-001",
    version="1.0.0",
    title="Comparison set",
    description=(
        "Members on their common window in the comparison currency: KPI matrix (best per row), "
        "base-100, drawdown, rolling 63-session vol and 252-session Sharpe overlays, calendar-year "
        "grid, correlation of daily returns on common sessions, R-016 stress windows and return "
        "quantiles. Each member uses its own annualisation unless harmonised to 252."
    ),
    applies_to=COMPARISON,
    params=CompareParams,
    result_kind="table",
    requires={"members"},
)
def comparison_set(inputs: AnalysisInputs, params: CompareParams) -> AnalysisOutput:
    members = _members(inputs)
    start, end = common_window(inputs, members)
    labels = [m["label"] for m in members]
    factors = [252 if params.harmonise else int(m.get("annualisation") or 252) for m in members]
    rf = float(inputs.conventions.get("rf") or 0.0)
    head: dict[str, Any] = {
        "window": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "days": int((end - start).days),
        },
        "currency": inputs.conventions.get("currency"),
        "members": labels,
        "annualisation": dict(zip(labels, factors, strict=True)),
    }
    notes: list[str] = []
    if len(set(factors)) > 1:
        notes.append(
            "Members use their own annualisation (crypto 365, markets 252); harmonise to "
            "compare at 252."
        )
    view = params.view
    if view == "kpi":
        b = _benchmark(members)
        bench = _clip(members[b]["series"], start, end) if b is not None else None
        per_member = [
            member_kpis(_clip(m["series"], start, end), f, rf, bench if i != b else None)
            for i, (m, f) in enumerate(zip(members, factors, strict=True))
        ]
        rows, grid_values, units, best, groups = [], [], [], [], []
        for group, metric, unit, direction in KPIS:
            row = [_f(v.get(metric)) for v in per_member]
            if all(x is None for x in row):
                continue
            ranked = [(x, i) for i, x in enumerate(row) if x is not None]
            pick = (
                (max(ranked)[1] if direction > 0 else min(ranked)[1])
                if direction and len(ranked) > 1
                else None
            )
            rows.append(metric)
            grid_values.append(row)
            units.append(unit)
            best.append(pick)
            groups.append(group)
        if b is None:
            notes.append(
                "Flag a member as benchmark (a benchmark preset or portfolio) for relative KPIs."
            )
        return AnalysisOutput(
            "table",
            head
            | {
                "grid": _grid(rows, labels, grid_values, units, best=best, groups=groups),
                "benchmark": labels[b] if b is not None else None,
            },
            tuple(notes),
        )
    frame = _union(members, start, end)
    if view in ("overlay", "drawdown", "rolling_vol", "rolling_sharpe"):
        if view == "overlay":
            out = 100 * frame / frame.iloc[0]
        elif view == "drawdown":
            out = frame / frame.cummax() - 1
        else:
            r = frame.pct_change()
            fac = pd.Series(factors, index=labels)
            if view == "rolling_vol":
                out = r.rolling(63).std() * np.sqrt(fac)
            else:
                ex = r - rf / fac
                out = ex.rolling(252).mean() / ex.rolling(252).std() * np.sqrt(fac)
        out = out.dropna(how="all")
        return AnalysisOutput(
            "series",
            head
            | {
                "dates": [d.date().isoformat() for d in out.index],
                "lines": {k: [_f(v) for v in out[k]] for k in out.columns},
            },
            tuple(notes),
        )
    if view == "calendar":
        year_end = frame.groupby(frame.index.year).last()
        prev = year_end.shift(1)
        prev.iloc[0] = frame.iloc[0]
        years = year_end / prev - 1
        return AnalysisOutput(
            "table",
            head
            | {
                "grid": _grid(
                    [str(y) for y in years.index],
                    labels,
                    [[_f(v) for v in row] for row in years.to_numpy()],
                    "percent",
                    note="first and last years are partial",
                )
            },
        )
    if view == "correlation":
        r = (
            pd.concat(
                {m["label"]: _clip(m["series"], start, end) for m in members}, axis=1, join="inner"
            )
            .pct_change()
            .dropna()
        )
        if len(r) < 20:
            raise InsufficientDataError("Too few common sessions for correlations")
        return AnalysisOutput(
            "table",
            head
            | {"labels": labels, "matrix": r.corr().to_numpy().tolist(), "observations": len(r)},
            ("Correlation on sessions common to all members (crypto weekends dropped).",),
        )
    if view == "stress":
        rows, values = [], []
        for name, s, e in STRESS_WINDOWS:
            ts, te = pd.Timestamp(s), pd.Timestamp(e)
            row = []
            for m in members:
                series = m["series"].dropna()
                covered = series.index[0] <= ts and series.index[-1] >= te
                row.append(_f(series.asof(te) / series.asof(ts) - 1) if covered else None)
            if any(v is not None for v in row):
                rows.append(name)
                values.append(row)
        if not rows:
            raise InsufficientDataError("No member covers a stress window")
        return AnalysisOutput(
            "table",
            head
            | {
                "grid": _grid(
                    rows, labels, values, "percent", note="full history, not the common window"
                )
            },
        )
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    r = frame.pct_change().dropna(how="all")
    values = [[_f(r[label].dropna().quantile(q)) for label in labels] for q in qs]
    return AnalysisOutput(
        "table", head | {"grid": _grid([f"p{int(q * 100)}" for q in qs], labels, values, "percent")}
    )


class RankParams(BaseModel):
    focus: str | None = Field(
        None, max_length=60, description="Member label (default: the first portfolio)"
    )


@analysis(
    id="CP-003",
    version="1.0.0",
    title="Peer ranking",
    description=(
        "Percentile rank of one member (default: the first portfolio) within the set for CAGR, "
        "volatility (lower is better), Sharpe and max drawdown on the common window."
    ),
    applies_to=COMPARISON,
    params=RankParams,
    result_kind="table",
    requires={"members"},
)
def peer_ranking(inputs: AnalysisInputs, params: RankParams) -> AnalysisOutput:
    members = _members(inputs)
    start, end = common_window(inputs, members)
    rf = float(inputs.conventions.get("rf") or 0.0)
    labels = [m["label"] for m in members]
    focus = params.focus or next(
        (m["label"] for m in members if m.get("kind") == "portfolio"), labels[0]
    )
    if focus not in labels:
        raise InsufficientDataError(f"{focus} is not a member ({', '.join(labels)})")
    kpis = {
        m["label"]: member_kpis(
            _clip(m["series"], start, end), int(m.get("annualisation") or 252), rf, None
        )
        for m in members
    }
    rows = []
    for metric, direction in (("cagr", 1), ("volatility", -1), ("sharpe", 1), ("max_drawdown", 1)):
        values = {k: x for k, v in kpis.items() if (x := v.get(metric)) is not None}
        if focus not in values or len(values) < 2:
            continue
        mine = values[focus]
        others = [v for k, v in values.items() if k != focus]
        beaten = sum(1 for v in others if (mine > v if direction > 0 else mine < v))
        rows.append(
            {
                "metric": metric,
                "value": mine,
                "rank": 1 + sum(1 for v in others if (v > mine if direction > 0 else v < mine)),
                "of": len(values),
                "percentile": beaten / len(others),
            }
        )
    return AnalysisOutput("table", {"focus": focus, "rows": rows, "members": labels})
