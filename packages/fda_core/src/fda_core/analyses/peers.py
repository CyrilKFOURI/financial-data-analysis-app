"""Peers & sector (PE-001…PE-007, module M12), earnings (E-001…E-005, M13) and ownership &
liquidity (O-001…O-006, M13).

Inputs: `data["peers"]` (stored group: members with D-PE-2 scores and provider metrics, weekly
subject/basket series), `data["company"]` (third-party snapshots with source and date),
`prices`, `other_prices` (benchmark), `fundamentals`. Third-party views (consensus, targets)
are shown as data with their source — never as the app's opinion.
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.analyses.price import _field, _prices, _windowed
from fda_core.analyses.statements import COMPANY, _facts, build_periods
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _group(inputs: AnalysisInputs) -> dict[str, Any]:
    group = inputs.data.get("peers")
    if not isinstance(group, dict) or not group.get("members"):
        raise InsufficientDataError("No peer group yet: build peers (auto) or add peers manually")
    return group


def _snap(inputs: AnalysisInputs, kind: str) -> dict[str, Any]:
    company = inputs.data.get("company")
    if not isinstance(company, dict) or kind not in company:
        raise InsufficientDataError(
            "No company data snapshot: ingest company data (earnings, holders…)"
        )
    snapshot: dict[str, Any] = company[kind]
    return snapshot


class NoParams(BaseModel):
    pass


# ── PE-001 / PE-002 peer group ───────────────────────────────────────────────


@analysis(
    id="PE-001",
    version="1.0.0",
    title="Auto peer group",
    description=(
        "D-PE-2: same-industry candidates within 0.25×–4× market cap, scored 0.4·industry depth + "
        "0.3·2-year weekly-return correlation + 0.2·size proximity (log) + 0.1·margin similarity; "
        "top 10 with score breakdown. Manual edits (PE-002) freeze the group until reset."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"peers"},
)
def peer_group(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    group = _group(inputs)
    members = [
        {
            "symbol": m["symbol"],
            "name": m["name"],
            "score": m["score"],
            "source": m["source"],
            "pinned": m["pinned"],
            **{f"score_{k}": v for k, v in (m.get("breakdown") or {}).items()},
            "market_cap": (m.get("metrics") or {}).get("marketCap"),
        }
        for m in group["members"]
    ]
    return AnalysisOutput(
        "table",
        {
            "mode": group["mode"],
            "algorithm_version": group["algorithm_version"],
            "weights": group["weights"],
            "built_at": group["created_at"],
            "members": members,
        },
    )


# ── PE-003 comps / PE-004 relative valuation / PE-005 benchmarking ───────────

COMPS = [
    ("trailingPE", "P/E", "x"),
    ("forwardPE", "P/E fwd", "x"),
    ("enterpriseToEbitda", "EV/EBITDA", "x"),
    ("enterpriseToRevenue", "EV/Sales", "x"),
    ("priceToBook", "P/B", "x"),
    ("grossMargins", "Gross margin", "pct"),
    ("operatingMargins", "Operating margin", "pct"),
    ("profitMargins", "Net margin", "pct"),
    ("revenueGrowth", "Revenue growth", "pct"),
    ("returnOnEquity", "ROE", "pct"),
    ("debtToEquity", "Debt / equity", "x"),
    ("marketCap", "Market cap", "amount"),
    ("dividendYield", "Dividend yield", "pct"),
]


def _peer_frame(group: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = {m["symbol"]: m.get("metrics") or {} for m in group["members"]}
    frame = pd.DataFrame(rows).T
    subject = (group.get("details") or {}).get("subject") or {}
    return frame, subject


@analysis(
    id="PE-003",
    version="1.0.0",
    title="Comps table",
    description=(
        "Subject vs peers on multiples, margins, growth, returns and leverage (provider metrics "
        "for all, so the comparison is like-for-like): peer median and mean, the subject's "
        "percentile rank and z-score per metric."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"peers"},
)
def comps(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    group = _group(inputs)
    frame, subject = _peer_frame(group)
    rows = []
    for key, label, unit in COMPS:
        values = pd.to_numeric(frame.get(key, pd.Series(dtype=float)), errors="coerce").dropna()
        own = _f(subject.get(key))
        row: dict[str, Any] = {
            "metric": label,
            "unit": unit,
            "subject": own,
            "peer_median": _f(values.median()) if len(values) else None,
            "peer_mean": _f(values.mean()) if len(values) else None,
            "n": len(values),
        }
        if own is not None and len(values) >= 3:
            row["percentile"] = float((values <= own).mean())
            std = values.std(ddof=1)
            row["z_score"] = _f((own - values.mean()) / std) if std else None
        rows.append(row)
    peers = [
        {"symbol": sym, **{k: _f(v) for k, v in metrics.items() if k != "currency"}}
        for sym, metrics in frame.to_dict("index").items()
    ]
    return AnalysisOutput("table", {"rows": rows, "peers": peers, "source": "yfinance"})


class RelativeParams(BaseModel):
    method: Literal["median", "regression"] = "median"


@analysis(
    id="PE-004",
    version="1.0.1",
    title="Relative valuation",
    description=(
        "Implied value per share from peer median multiples applied to the subject's TTM figures "
        "(P/E × EPS, EV/EBITDA and EV/Sales through the EV bridge, P/B × book); cross-sectional "
        "regression of P/E on revenue growth across peers gives a growth-adjusted P/E."
    ),
    applies_to=COMPANY,
    params=RelativeParams,
    result_kind="table",
    requires={"peers", "fundamentals", "prices"},
)
def relative_valuation(inputs: AnalysisInputs, params: RelativeParams) -> AnalysisOutput:
    from fda_core.analyses.valuation import Snapshot

    group = _group(inputs)
    frame, subject = _peer_frame(group)
    s = Snapshot(inputs)
    med = {
        k: _f(pd.to_numeric(frame.get(k, pd.Series(dtype=float)), errors="coerce").median())
        for k in ("trailingPE", "enterpriseToEbitda", "enterpriseToRevenue", "priceToBook")
    }
    ni, ebitda, revenue = (
        s.flow_value("net_income"),
        s.flow_value("ebitda"),
        s.flow_value("revenue"),
    )
    equity = s.stock_value("total_equity")
    bridge = s.net_debt + s.minorities + s.preferred
    implied = {
        "pe": med["trailingPE"] * ni / s.shares if med["trailingPE"] and ni > 0 else None,
        "ev_ebitda": (med["enterpriseToEbitda"] * ebitda - bridge) / s.shares
        if med["enterpriseToEbitda"] and ebitda > 0
        else None,
        "ev_sales": (med["enterpriseToRevenue"] * revenue - bridge) / s.shares
        if med["enterpriseToRevenue"] and revenue > 0
        else None,
        "pb": med["priceToBook"] * equity / s.shares if med["priceToBook"] and equity > 0 else None,
    }
    regression = None
    pe = pd.to_numeric(frame.get("trailingPE"), errors="coerce")
    growth = pd.to_numeric(frame.get("revenueGrowth"), errors="coerce")
    both = pd.concat([growth, pe], axis=1).dropna()
    both = both[(both.iloc[:, 1] > 0) & (both.iloc[:, 1] < 200)]
    own_growth = _f(subject.get("revenueGrowth"))
    if len(both) >= 4 and own_growth is not None:
        slope, intercept = np.polyfit(both.iloc[:, 0], both.iloc[:, 1], 1)
        fitted_pe = intercept + slope * own_growth
        corr = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        regression = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": corr**2,
            "n": len(both),
            "fitted_pe": float(fitted_pe),
            "implied_per_share": float(fitted_pe * ni / s.shares)
            if ni > 0 and fitted_pe > 0
            else None,
        }
    return AnalysisOutput(
        "table",
        {
            "price": s.price,
            "peer_medians": med,
            "implied_per_share": implied,
            "regression": regression,
            "basis": s.basis,
            "currency": s.currency,
        },
        (
            "Peer multiples from the provider; subject figures from filings (TTM). Model "
            "output, not a target.",
        ),
    )


@analysis(
    id="PE-005",
    version="1.0.0",
    title="Operating benchmarking",
    description=(
        "Scatter maps of subject and peers: ROE vs revenue growth, EV/EBITDA vs operating margin, "
        "P/E vs growth."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="series",
    requires={"peers"},
)
def benchmarking(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    group = _group(inputs)
    frame, subject = _peer_frame(group)
    maps = {}
    for name, x, y in (
        ("roe_vs_growth", "revenueGrowth", "returnOnEquity"),
        ("ev_ebitda_vs_margin", "operatingMargins", "enterpriseToEbitda"),
        ("pe_vs_growth", "revenueGrowth", "trailingPE"),
    ):
        points = [
            {"symbol": sym, "x": _f(row.get(x)), "y": _f(row.get(y))}
            for sym, row in frame.to_dict("index").items()
        ]
        maps[name] = {
            "x": x,
            "y": y,
            "peers": [p for p in points if p["x"] is not None and p["y"] is not None],
            "subject": {"x": _f(subject.get(x)), "y": _f(subject.get(y))},
        }
    return AnalysisOutput("series", {"maps": maps})


# Sector ETFs (D-PE-4): SPDR for the US.
SECTOR_ETF = {
    "technology": "XLK",
    "financial-services": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "industrials": "XLI",
    "consumer-cyclical": "XLY",
    "consumer-defensive": "XLP",
    "utilities": "XLU",
    "basic-materials": "XLB",
    "real-estate": "XLRE",
    "communication-services": "XLC",
}


@analysis(
    id="PE-006",
    version="1.0.0",
    title="Sector dashboard",
    description=(
        "Peer-group aggregates as the sector view (median margins, growth, ROE and multiples, "
        "interquartile range) and the sector ETF (D-PE-4) mapped from the provider sector."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"peers"},
)
def sector_dashboard(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    group = _group(inputs)
    frame, subject = _peer_frame(group)
    rows = []
    for key, label, unit in COMPS:
        values = pd.to_numeric(frame.get(key, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(values) < 3:
            continue
        rows.append(
            {
                "metric": label,
                "unit": unit,
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q3": float(values.quantile(0.75)),
                "subject": _f(subject.get(key)),
            }
        )
    sector_key = (group.get("details") or {}).get("industry_key")
    etf = SECTOR_ETF.get(str(inputs.conventions.get("sector") or "").lower().replace(" ", "-"))
    return AnalysisOutput(
        "table", {"industry": sector_key, "sector_etf": etf, "rows": rows, "peers": len(frame)}
    )


@analysis(
    id="PE-007",
    version="1.0.0",
    title="Peer price behaviour",
    description=(
        "Over the last two years of weekly data: subject vs equal-weight peer basket cumulative "
        "performance, correlation, beta to the basket and each peer's return correlation (D-PE-2)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="series",
    requires={"peers"},
)
def peer_price_behaviour(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    group = _group(inputs)
    weekly = (group.get("details") or {}).get("weekly") or {}
    subject = pd.Series(weekly.get("subject") or [], dtype=float)
    basket_ret = pd.Series(weekly.get("basket_returns") or [], dtype=float)
    if len(subject) < 30 or len(basket_ret) != len(subject):
        raise InsufficientDataError("Weekly peer data unavailable: rebuild the auto peer group")
    subject_ret = subject.pct_change()
    both = pd.concat([subject_ret, basket_ret], axis=1).dropna()
    beta = float(
        np.cov(both.iloc[:, 0], both.iloc[:, 1], ddof=1)[0, 1] / both.iloc[:, 1].var(ddof=1)
    )
    return AnalysisOutput(
        "series",
        {
            "dates": weekly.get("dates"),
            "subject_index": [
                None if not np.isfinite(v) else float(v)
                for v in (1 + subject_ret.fillna(0)).cumprod() * 100
            ],
            "basket_index": [
                None if not np.isfinite(v) else float(v)
                for v in (1 + basket_ret.fillna(0)).cumprod() * 100
            ],
            "correlation": float(both.corr().iloc[0, 1]),
            "beta_to_basket": beta,
            "peer_correlations": {
                m["symbol"]: (m.get("breakdown") or {}).get("correlation") for m in group["members"]
            },
        },
    )


# ── E-001…E-005 earnings & estimates ─────────────────────────────────────────


def _earnings(inputs: AnalysisInputs) -> pd.DataFrame:
    snap = _snap(inputs, "earnings_dates")
    frame = pd.DataFrame(snap["payload"])
    if frame.empty:
        raise InsufficientDataError("No earnings dates from the provider")
    date_col = next((c for c in frame.columns if "Date" in c), "index")
    frame["date"] = pd.to_datetime(frame[date_col], utc=True)
    frame = frame.rename(
        columns={
            "EPS Estimate": "estimate",
            "Reported EPS": "reported",
            "Surprise(%)": "surprise_pct",
        }
    )
    return frame.sort_values("date")


@analysis(
    id="E-001",
    version="1.0.0",
    title="Earnings calendar & history",
    description=(
        "Past and next report dates with EPS estimate, reported EPS and surprise (third-party "
        "data, source and date shown)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def earnings_calendar(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    frame = _earnings(inputs)
    snap = _snap(inputs, "earnings_dates")
    now = pd.Timestamp.now(tz="UTC")
    upcoming = frame[frame["date"] > now]
    rows = [
        {"date": d.isoformat(), "estimate": _f(e), "reported": _f(r), "surprise_pct": _f(s)}
        for d, e, r, s in zip(
            frame["date"],
            frame.get("estimate"),
            frame.get("reported"),
            frame.get("surprise_pct"),
            strict=True,
        )
    ]
    return AnalysisOutput(
        "table",
        {
            "next_report": upcoming["date"].iloc[0].isoformat() if len(upcoming) else None,
            "history": rows[::-1],
            "source": snap["source"],
            "as_of": snap["as_of"],
        },
    )


@analysis(
    id="E-002",
    version="1.0.0",
    title="Surprise statistics",
    description=(
        "Over reported quarters: beat rate (reported > estimate), mean and median surprise %, "
        "and SUE = (reported − estimate) / std of past surprises (expanding, no look-ahead)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def surprise_stats(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    frame = _earnings(inputs).dropna(subset=["reported", "estimate"])
    if len(frame) < 4:
        raise InsufficientDataError("Fewer than four reported quarters")
    surprise = frame["reported"] - frame["estimate"]
    sue = surprise / surprise.expanding().std().shift(1)
    return AnalysisOutput(
        "table",
        {
            "quarters": len(frame),
            "beat_rate": float((surprise > 0).mean()),
            "mean_surprise_pct": _f(frame["surprise_pct"].mean()),
            "median_surprise_pct": _f(frame["surprise_pct"].median()),
            "last_sue": _f(sue.iloc[-1]),
            "sue": [
                {"date": d.isoformat(), "sue": _f(v)}
                for d, v in zip(frame["date"], sue, strict=True)
            ],
        },
    )


class EventParams(BaseModel):
    estimation_window: int = Field(250, ge=60, le=500)
    gap: int = Field(30, ge=5, le=60)
    pre: int = Field(5, ge=1, le=20)
    post: int = Field(20, ge=5, le=60)


@analysis(
    id="E-003",
    version="1.0.0",
    title="Earnings event study",
    description=(
        "Market model r = α + β·r_m estimated on [−estimation − gap, −gap] sessions before each "
        "report; abnormal returns over [−pre, +post]; average cumulative abnormal return path, "
        "per-event CAR and day-0 gap (after-close reports count from the next session)."
    ),
    applies_to=COMPANY,
    params=EventParams,
    result_kind="series",
    requires={"company", "prices", "other_prices"},
)
def event_study(inputs: AnalysisInputs, params: EventParams) -> AnalysisOutput:
    frame = _earnings(inputs).dropna(subset=["reported"])
    prices = inputs.data.get("prices")
    bench = inputs.data.get("other_prices")
    if not isinstance(bench, pd.DataFrame) or bench.empty or not isinstance(prices, pd.DataFrame):
        raise InsufficientDataError(
            f"Ingest the benchmark {inputs.conventions.get('other_symbol')} for abnormal returns"
        )
    r = pd.concat(
        [prices[_field(inputs)].pct_change(), bench[_field(inputs)].pct_change()],
        axis=1,
        join="inner",
    ).dropna()
    r.columns = ["asset", "market"]
    sessions = r.index
    window = range(-params.pre, params.post + 1)
    paths, events = [], []
    for when in frame["date"]:
        local = when.tz_convert("America/New_York") if when.tzinfo else when
        day0 = pd.Timestamp(local.date()) + (
            pd.Timedelta(days=1) if local.hour >= 16 else pd.Timedelta(0)
        )
        i0 = int(sessions.searchsorted(day0))
        start = i0 - params.gap - params.estimation_window
        if start < 0 or i0 + params.post >= len(sessions):
            continue
        est = r.iloc[start : i0 - params.gap]
        beta = float(np.cov(est["asset"], est["market"], ddof=1)[0, 1] / est["market"].var(ddof=1))
        alpha = float(est["asset"].mean() - beta * est["market"].mean())
        ev = r.iloc[i0 - params.pre : i0 + params.post + 1]
        ar = ev["asset"] - (alpha + beta * ev["market"])
        paths.append(ar.cumsum().to_numpy())
        events.append(
            {
                "date": sessions[i0].date().isoformat(),
                "car": float(ar.sum()),
                "day0_ar": float(ar.iloc[params.pre]),
                "car_0_1": float(ar.iloc[params.pre : params.pre + 2].sum()),
            }
        )
    if len(paths) < 3:
        raise InsufficientDataError("Fewer than three reports with enough surrounding history")
    mean_path = np.mean(paths, axis=0)
    se = np.std(paths, axis=0, ddof=1) / math.sqrt(len(paths))
    return AnalysisOutput(
        "series",
        {
            "days": list(window),
            "mean_car": mean_path.tolist(),
            "se": se.tolist(),
            "events": events,
            "n_events": len(events),
            "benchmark": inputs.conventions.get("other_symbol"),
        },
    )


@analysis(
    id="E-004",
    version="1.0.0",
    title="Estimate trends",
    description=(
        "Consensus EPS now vs 7/30/60/90 days ago and revision counts (up/down, 7/30 days) per "
        "horizon; third-party data."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def estimate_trends(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    trend = _snap(inputs, "eps_trend")
    out: dict[str, Any] = {
        "trend": trend["payload"],
        "source": trend["source"],
        "as_of": trend["as_of"],
    }
    company = inputs.data.get("company") or {}
    for kind in ("eps_revisions", "earnings_estimate", "revenue_estimate"):
        if kind in company:
            out[kind] = company[kind]["payload"]
    return AnalysisOutput("table", out)


@analysis(
    id="E-005",
    version="1.0.0",
    title="Analyst views (third-party)",
    description=(
        "Consensus rating distribution over the last months and price-target range, shown as "
        "provider data with source and date (never the app's opinion)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def analyst_views(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    recs = _snap(inputs, "recommendations_summary")
    company = inputs.data.get("company") or {}
    targets = company.get("price_targets", {}).get("payload")
    return AnalysisOutput(
        "table",
        {
            "ratings": recs["payload"],
            "targets": targets,
            "source": recs["source"],
            "as_of": recs["as_of"],
        },
        ("Third-party consensus, displayed as data; it is not an opinion of this application.",),
    )


# ── O-001…O-006 ownership & liquidity ────────────────────────────────────────


@analysis(
    id="O-001",
    version="1.0.0",
    title="Holder breakdown",
    description=(
        "Insiders %, institutions %, float and top institutional holders (provider data, latest "
        "13F aggregation)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def holder_breakdown(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    info = _snap(inputs, "info")["payload"]
    company = inputs.data.get("company") or {}
    holders = company.get("institutional_holders", {}).get("payload") or []
    return AnalysisOutput(
        "table",
        {
            "insiders_pct": _f(info.get("heldPercentInsiders")),
            "institutions_pct": _f(info.get("heldPercentInstitutions")),
            "float_shares": _f(info.get("floatShares")),
            "shares_outstanding": _f(info.get("sharesOutstanding")),
            "top_holders": holders[:10],
        },
    )


@analysis(
    id="O-002",
    version="1.0.0",
    title="Institutional changes",
    description=(
        "Quarter-on-quarter change in shares held by the top institutional holders (13F, US)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def institutional_changes(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    holders = _snap(inputs, "institutional_holders")["payload"]
    rows = [
        {
            "holder": h.get("Holder"),
            "reported": h.get("Date Reported"),
            "pct_held": _f(h.get("pctHeld")),
            "pct_change": _f(h.get("pctChange")),
            "shares": _f(h.get("Shares")),
        }
        for h in holders
    ]
    changes = [r["pct_change"] for r in rows if r["pct_change"] is not None]
    return AnalysisOutput(
        "table",
        {
            "holders": rows,
            "net_direction": "adding"
            if changes and np.mean(changes) > 0
            else "reducing"
            if changes
            else None,
        },
    )


@analysis(
    id="O-003",
    version="1.0.0",
    title="Insider activity",
    description=(
        "Form 4 transactions (via provider): purchases vs sales, net insider value over 3/6/12 "
        "months, cluster buying (≥ 3 distinct insiders buying within 30 days)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def insider_activity(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    frame = pd.DataFrame(_snap(inputs, "insider_transactions")["payload"])
    if frame.empty:
        raise InsufficientDataError("No insider transactions reported")
    frame["date"] = pd.to_datetime(frame.get("Start Date"), errors="coerce")
    text = frame.get("Text", pd.Series("", index=frame.index)).fillna("").str.lower()
    frame["side"] = np.where(
        text.str.contains("purchase|buy"),
        "buy",
        np.where(text.str.contains("sale|sell"), "sell", "other"),
    )
    frame["value"] = pd.to_numeric(frame.get("Value"), errors="coerce").fillna(0)
    now = frame["date"].max()
    net = {}
    for months in (3, 6, 12):
        recent = frame[frame["date"] >= now - pd.DateOffset(months=months)]
        net[f"{months}m"] = float(
            recent.loc[recent["side"] == "buy", "value"].sum()
            - recent.loc[recent["side"] == "sell", "value"].sum()
        )
    buys = frame[frame["side"] == "buy"].sort_values("date")
    cluster = (
        any(
            buys[(buys["date"] >= d) & (buys["date"] <= d + pd.Timedelta(days=30))][
                "Insider"
            ].nunique()
            >= 3
            for d in buys["date"]
        )
        if len(buys)
        else False
    )
    return AnalysisOutput(
        "table",
        {
            "net_value": net,
            "cluster_buying": bool(cluster),
            "transactions": [
                {
                    "date": d.date().isoformat() if pd.notna(d) else None,
                    "insider": i,
                    "position": p,
                    "side": s,
                    "shares": _f(sh),
                    "value": _f(v),
                }
                for d, i, p, s, sh, v in zip(
                    frame["date"],
                    frame.get("Insider"),
                    frame.get("Position"),
                    frame["side"],
                    frame.get("Shares"),
                    frame["value"],
                    strict=True,
                )
            ][:40],
        },
    )


@analysis(
    id="O-004",
    version="1.0.0",
    title="Short interest",
    description=(
        "Shares short, % of float, days to cover (short ratio) and settlement date (provider-"
        "dependent, US)."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"company"},
)
def short_interest(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    info = _snap(inputs, "info")["payload"]
    if info.get("sharesShort") is None:
        raise InsufficientDataError("The provider reports no short interest for this listing")
    settle = info.get("dateShortInterest")
    return AnalysisOutput(
        "table",
        {
            "shares_short": _f(info.get("sharesShort")),
            "pct_float": _f(info.get("shortPercentOfFloat")),
            "days_to_cover": _f(info.get("shortRatio")),
            "settlement_date": pd.Timestamp(settle, unit="s").date().isoformat()
            if isinstance(settle, (int, float))
            else settle,
        },
    )


@analysis(
    id="O-005",
    version="1.0.0",
    title="Share count & dilution",
    description=(
        "Basic/diluted weighted shares and period-end shares by fiscal year, annual dilution rate,"
        " SBC as % of revenue and gross buybacks vs SBC."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires={"fundamentals"},
)
def dilution(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    wide = build_periods(_facts(inputs), "FY").iloc[:, -10:]

    def row(code: str) -> pd.Series:
        return (
            wide.loc[code].astype(float)
            if code in wide.index
            else pd.Series(np.nan, index=wide.columns)
        )

    diluted = row("shares_diluted_wavg")
    return AnalysisOutput(
        "table",
        {
            "periods": list(wide.columns),
            "rows": [
                {
                    "code": "shares_diluted",
                    "label": "Diluted shares (w. avg)",
                    "unit": "count",
                    "values": [_f(v) for v in diluted],
                },
                {
                    "code": "shares_outstanding",
                    "label": "Shares outstanding",
                    "unit": "count",
                    "values": [_f(v) for v in row("shares_outstanding")],
                },
                {
                    "code": "dilution",
                    "label": "Net dilution (−: buyback)",
                    "unit": "pct",
                    "values": [_f(v) for v in diluted.pct_change()],
                },
                {
                    "code": "sbc_pct_revenue",
                    "label": "SBC % revenue",
                    "unit": "pct",
                    "values": [_f(v) for v in row("sbc") / row("revenue")],
                },
                {
                    "code": "buybacks_to_sbc",
                    "label": "Buybacks / SBC",
                    "unit": "x",
                    "values": [_f(v) for v in row("buybacks") / row("sbc")],
                },
            ],
        },
    )


class LiquidityParams(BaseModel):
    position_value: float = Field(
        1_000_000, gt=0, le=1e11, description="Position to trade (currency)"
    )
    participation: float = Field(0.10, gt=0, le=1, description="Max share of ADV per day")


@analysis(
    id="O-006",
    version="1.0.0",
    title="Liquidity profile",
    description=(
        "ADV over 20/60 sessions in shares and value, turnover (volume / shares), Amihud "
        "illiquidity, Corwin–Schultz spread, and days to trade a position at a participation "
        "rate of ADV."
    ),
    applies_to={"asset"},
    params=LiquidityParams,
    result_kind="table",
    requires={"prices"},
)
def liquidity_profile(inputs: AnalysisInputs, params: LiquidityParams) -> AnalysisOutput:
    from fda_core.analyses.ts_info import corwin_schultz

    if not inputs.conventions.get("has_volume", True):
        raise InsufficientDataError("No meaningful volume for this asset type")
    full = _prices(inputs)
    frame = _windowed(inputs, full)
    close, volume = frame["close"], frame["volume"]
    value = close * volume
    adv20, adv60 = float(volume.iloc[-20:].mean()), float(volume.iloc[-60:].mean())
    adv_value = float(value.iloc[-20:].mean())
    amihud = float((close.pct_change().abs() / value.replace(0, np.nan)).iloc[-60:].mean() * 1e6)
    spread = float(corwin_schultz(full["high"], full["low"]).iloc[-60:].mean())
    return AnalysisOutput(
        "table",
        {
            "adv_20d_shares": adv20,
            "adv_60d_shares": adv60,
            "adv_20d_value": adv_value,
            "amihud_per_million": amihud,
            "corwin_schultz_spread": spread,
            "days_to_trade": params.position_value / (params.participation * adv_value)
            if adv_value
            else None,
            "position_value": params.position_value,
            "participation": params.participation,
        },
    )
