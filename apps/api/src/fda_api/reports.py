"""Report builder (CP-004, M17): templates of analyses rendered as one self-contained HTML
document — Zero-Slop print stylesheet (light, 1 px rules, tabular numbers), inline SVG charts,
and an appendix with every analysis' formula, version, data version and the conventions used.
PDF = the browser's print of this page (A4)."""

import html
import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine

from fda_api.runner import Run, run_analysis
from fda_data.lake import PriceLake

TEMPLATES: dict[str, dict[str, Any]] = {
    "company": {
        "title": "Company tear sheet",
        "subject": "asset",
        "analyses": [
            ("T-005", {}),
            ("T-007", {}),
            ("R-020", {}),
            ("T-002", {}),
            ("R-001", {}),
            ("F-001", {}),
            ("F-011", {}),
            ("F-012", {}),
            ("S-001", {}),
            ("V-001", {}),
            ("V-016", {}),
            ("PE-003", {}),
            ("R-010", {}),
        ],
    },
    "risk": {
        "title": "Security risk sheet",
        "subject": "asset",
        "analyses": [
            ("R-020", {}),
            ("T-015", {}),
            ("T-014", {}),
            ("R-010", {}),
            ("R-011", {}),
            ("R-001", {}),
            ("R-002", {}),
            ("R-016", {}),
            ("TS-003", {}),
            ("TS-010", {}),
        ],
    },
    "portfolio": {
        "title": "Portfolio review",
        "subject": "portfolio",
        "analyses": [
            ("P-003", {}),
            ("P-004", {}),
            ("R-020", {}),
            ("P-008", {}),
            ("P-018", {}),
            ("P-009", {}),
            ("P-010", {}),
            ("P-006", {}),
            ("P-019", {}),
            ("P-022", {}),
            ("P-023", {}),
            ("P-005", {}),
        ],
    },
    "comparison": {
        "title": "Comparison",
        "subject": "comparison",
        "analyses": [
            ("CP-001", {"view": "kpi"}),
            ("CP-001", {"view": "overlay"}),
            ("CP-001", {"view": "drawdown"}),
            ("CP-001", {"view": "calendar"}),
            ("CP-001", {"view": "correlation"}),
        ],
    },
    "ips": {
        "title": "Investment policy statement",
        "subject": "profile",
        "note": "Model output for the owner's own use — not personalised advice for third parties.",
        "analyses": [("IN-001", {}), ("IN-003", {}), ("IN-002", {})],
        "policy": True,
    },
}

CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font: 10.5px/1.45 Inter, "Segoe UI", system-ui, sans-serif; color: #111; background: #fff;
       margin: 0 auto; max-width: 190mm; font-variant-numeric: tabular-nums; }
h1 { font-size: 18px; margin: 0 0 2px; font-weight: 600; }
h2 { font-size: 12px; margin: 16px 0 4px; font-weight: 600; border-bottom: 1px solid #111;
     padding-bottom: 2px; break-after: avoid; }
.meta, .muted { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 2px 0 6px; break-inside: avoid; }
th, td { border-bottom: 1px solid #d4d4d4; padding: 1px 4px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 500; color: #555; text-transform: uppercase; font-size: 9px; letter-spacing: .04em; }
.neg { color: #b42318; }
.best { text-decoration: underline; }
svg { display: block; width: 100%; height: 110px; border-bottom: 1px solid #d4d4d4; }
.note { border-left: 1px solid #b54708; padding-left: 6px; color: #555; }
section { break-inside: avoid-page; }
.appendix td, .appendix th { text-align: left; vertical-align: top; }
"""

COLORS = ["#2563eb", "#111111", "#b54708", "#0f766e", "#7c3aed", "#be185d", "#4d7c0f", "#475569"]
PERCENT_HINTS = (
    "return",
    "vol",
    "drawdown",
    "cagr",
    "var",
    "es",
    "weight",
    "contribution",
    "twr",
    "mwr",
    "margin",
    "yield",
    "growth",
    "share",
    "excess",
    "allocation",
    "selection",
    "interaction",
    "total",
    "impact",
    "roe",
    "roic",
)


def _is_percent(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in PERCENT_HINTS) and not any(
        x in k for x in ("ratio", "beta", "days", "count", "t_")
    )


def _fmt(value: Any, key: str = "", unit: str | None = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        v = float(value)
        if not math.isfinite(v):
            return "—"
        if unit == "percent" or (unit is None and _is_percent(key) and abs(v) <= 50):
            return f"{v * 100:,.2f} %".replace("-", "−")
        if abs(v) >= 1e5:
            return f"{v:,.0f}".replace("-", "−")
        return f"{v:,.4g}".replace("-", "−") if abs(v) < 1 else f"{v:,.2f}".replace("-", "−")
    return html.escape(str(value).replace("_", " "))


def _cls(value: Any) -> str:
    return (
        ' class="neg"'
        if isinstance(value, int | float) and not isinstance(value, bool) and value < 0
        else ""
    )


def _scalars(payload: dict[str, Any], title: str = "") -> str:
    rows = [
        (title if k == "value" and title else k, v)
        for k, v in payload.items()
        if isinstance(v, int | float | str | bool) or v is None
    ]
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{_fmt(k)}</td><td{_cls(v)}>{_fmt(v, k)}</td></tr>" for k, v in rows[:24]
    )
    return f"<table>{body}</table>"


def _periods_table(key: str, periods: list[Any], rows: list[dict[str, Any]]) -> str:
    """Statement/ratio rows carrying a `values` list aligned to `periods`: last 5 periods."""
    keep = list(range(len(periods)))[-5:]
    head = "<th></th>" + "".join(f"<th>{html.escape(str(periods[i]))}</th>" for i in keep)
    body = ""
    for r in rows[:40]:
        values = r.get("values") or []
        unit = "percent" if r.get("unit") == "pct" else "number"
        label = r.get("label") or r.get("code") or r.get("metric") or ""
        cells = "".join(
            f"<td{_cls(v)}>{_fmt(v, unit=unit)}</td>"
            for v in (values[i] if i < len(values) else None for i in keep)
        )
        body += f"<tr><td>{_fmt(label)}</td>{cells}</tr>"
    return f'<p class="muted">{_fmt(key)}</p><table><tr>{head}</tr>{body}</table>'


def _records(key: str, rows: list[dict[str, Any]]) -> str:
    cols = [c for c in rows[0] if any(isinstance(r.get(c), int | float | str | bool) for r in rows)]
    if not cols:
        return ""
    head = "".join(f"<th>{_fmt(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td{_cls(r.get(c))}>{_fmt(r.get(c), c)}</td>" for c in cols) + "</tr>"
        for r in rows[:30]
    )
    return f'<p class="muted">{_fmt(key)}</p><table><tr>{head}</tr>{body}</table>'


def _grid(grid: dict[str, Any]) -> str:
    cols, units, best = grid["columns"], grid.get("unit"), grid.get("best") or []
    head = "<th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for i, (label, values) in enumerate(zip(grid["rows"], grid["values"], strict=False)):
        unit = units[i] if isinstance(units, list) else units
        cells = "".join(
            f"<td{' class=best' if i < len(best) and best[i] == j else ''}{_cls(v)}>{_fmt(v, unit=unit)}</td>"
            for j, v in enumerate(values)
        )
        body += f"<tr><td>{_fmt(label)}</td>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def _matrix(labels: list[str], matrix: list[list[float]]) -> str:
    return _grid({"rows": labels, "columns": labels, "values": matrix, "unit": "number"})


def _chart(dates: list[str], lines: dict[str, list[float | None]]) -> str:
    """Inline SVG polyline chart (≤ 8 lines), shared y-scale, first/last date labels."""
    series = list(lines.items())[:8]
    values = [v for _, vs in series for v in vs if v is not None and math.isfinite(v)]
    if len(dates) < 2 or not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    w, h = 600.0, 100.0
    paths = []
    for i, (name, vs) in enumerate(series):
        pts = [
            f"{w * k / (len(vs) - 1):.1f},{h - (v - lo) / span * h:.1f}"
            for k, v in enumerate(vs)
            if v is not None and math.isfinite(v)
        ]
        paths.append(
            f'<polyline fill="none" stroke="{COLORS[i]}" stroke-width="1" points="{" ".join(pts)}"><title>{html.escape(name)}</title></polyline>'
        )
    legend = " · ".join(
        f'<span style="color:{COLORS[i]}">{html.escape(n.replace("_", " "))}</span>'
        for i, (n, _) in enumerate(series)
    )
    return (
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none">{"".join(paths)}</svg>'
        f'<p class="muted">{dates[0]} → {dates[-1]} · {_fmt(lo)} … {_fmt(hi)} · {legend}</p>'
    )


def _series(payload: dict[str, Any]) -> str:
    for container in [payload, *[v for v in payload.values() if isinstance(v, dict)]]:
        dates = container.get("dates")
        if not isinstance(dates, list) or len(dates) < 2:
            continue
        lines = {
            k: v
            for k, v in container.items()
            if k != "dates"
            and isinstance(v, list)
            and len(v) == len(dates)
            and all(x is None or isinstance(x, int | float) for x in v)
        }
        nested = container.get("lines")
        if isinstance(nested, dict):
            lines |= {
                k: v for k, v in nested.items() if isinstance(v, list) and len(v) == len(dates)
            }
        if lines:
            return _chart(dates, lines)
    return ""


def render_run(run: Run) -> str:
    title = f"{html.escape(run.title)} <span class=muted>({run.analysis_id})</span>"
    if run.payload is None:
        return f'<section><h2>{title}</h2><p class="note">{html.escape(run.problem or "No result")}</p></section>'
    p = run.payload
    parts = [_series(p), _scalars(p, run.title)]
    if isinstance(p.get("grid"), dict):
        parts.append(_grid(p["grid"]))
    if isinstance(p.get("labels"), list) and isinstance(p.get("matrix"), list):
        parts.append(_matrix(p["labels"], p["matrix"]))
    for key, value in p.items():
        if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
            periods = p.get("periods")
            if isinstance(periods, list) and any(isinstance(r.get("values"), list) for r in value):
                parts.append(_periods_table(key, periods, value))
            else:
                parts.append(_records(key, value))
    parts += [f'<p class="note">{html.escape(w)}</p>' for w in run.warnings]
    return f"<section><h2>{title}</h2>{''.join(parts)}</section>"


def _policy(runs: list[Run]) -> str:
    """IPS sections from the profile and allocation payloads (M22 §5)."""
    by_id = {r.analysis_id: r.payload or {} for r in runs}
    prof, alloc = by_id.get("IN-001", {}), by_id.get("IN-003", {})
    goal = prof.get("goal") or {}
    rows = [
        ("Objective", f"Reach {_fmt(goal.get('amount'))} in {_fmt(goal.get('years'))} years"
         if goal else "Long-term growth consistent with the risk score"),
        ("Required return", _fmt(prof.get("required_return"), "return")),
        ("Risk score", f"{prof.get('risk_score')} (capacity {prof.get('capacity')}, willingness {prof.get('willingness')})"),
        ("Drawdown tolerance", _fmt(prof.get("max_drawdown_tolerance"), "drawdown")),
        ("Constraints", f"Horizon {html.escape(str((prof.get('rows') or [{}])[0].get('answer', '—')))}; "
         f"wrapper {html.escape(str(prof.get('wrapper') or 'not stated'))}; liquidity per questionnaire"),
        ("Strategic allocation", f"Score {alloc.get('risk_score')} model allocation; core share {_fmt(alloc.get('core_share'), 'share')}"),
        ("Ranges", "± 5 percentage points around each target weight"),
        ("Benchmark", "The strategic allocation itself, rebalanced quarterly"),
        ("Rebalancing", "Quarterly calendar review; trade back to target when a weight leaves its ± 5 pt range"),
        ("Review", "Annually, or after a change in situation, goal or risk tolerance"),
    ]  # fmt: skip
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in rows)
    return (
        f"<section><h2>Policy</h2><table>{body}</table>"
        "<p class='muted'>Signatures: ______________________ (investor) · date ____________</p></section>"
    )


def build_report(
    engine: Engine,
    lake: PriceLake,
    template: str,
    subject_id: str,
    context: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    spec = TEMPLATES[template]
    memo: dict[Any, Any] = {}
    overrides = overrides or {}
    runs = [
        run_analysis(
            engine,
            lake,
            spec["subject"],
            subject_id,
            aid,
            params | overrides.get(aid, {}),
            context,
            memo,
        )
        for aid, params in spec["analyses"]
    ]
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ctx = context or {}
    appendix = "".join(
        f"<tr><td>{r.analysis_id} v{r.version}</td><td>{html.escape(r.title)}</td>"
        f"<td>{html.escape(r.description)}</td><td>{html.escape(r.data_version or r.problem or '')}</td></tr>"
        for r in runs
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(spec["title"])} — {html.escape(subject_id)}</title><style>{CSS}</style></head>
<body>
<header><h1>{html.escape(spec["title"])} · {html.escape(subject_id)}</h1>
<p class="meta">Generated {now} · range {html.escape(str(ctx.get("range", "1Y")))} · currency {html.escape(str(ctx.get("ccy", "listing")))} ·
{html.escape(spec.get("note", "measures and explanations only, no investment advice."))}</p></header>
{_policy(runs) if spec.get("policy") else ""}
{"".join(render_run(r) for r in runs)}
<section class="appendix"><h2>Sources, formulas &amp; versions</h2>
<table><tr><th>Analysis</th><th>Title</th><th>Formula / method</th><th>Data version</th></tr>{appendix}</table>
<p class="muted">Prices: stored provider series (yfinance, cache-first); fundamentals: SEC companyfacts / provider statements, point-in-time.
Conventions come from each asset type (calendar, annualisation, price field).</p></section>
</body></html>"""
