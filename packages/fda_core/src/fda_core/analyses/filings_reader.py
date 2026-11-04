"""Filings reader (ANALYTICS_CATALOG AI-004, deterministic part; module M30 §5).

Input `filings` = EDGAR index + cached documents (extracted text) for the asset. Outputs the
filing list, the item sections of the two latest annual reports with their size change, an
excerpt of the chosen section, and the year-on-year risk-factor diff. Model-written summaries
with document citations need the agent runtime (J25) and are reported as pending.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from fda_core.filings import risk_blocks, risk_diff, split_sections
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

ANNUAL = ("10-K", "10-K/A", "20-F", "40-F")


class FilingsParams(BaseModel):
    section: str = Field("1A", max_length=3, pattern=r"^[0-9]{1,2}[A-C]?$", description="Item")
    excerpt_chars: int = Field(3000, ge=200, le=20_000)
    form: Literal["annual", "10-Q"] = "annual"


@analysis(
    id="AI-004",
    version="1.0.0",
    title="Filings reader",
    description=(
        "Latest SEC filings (10-K, 10-Q, 20-F, 8-K) with links; item sections of the two latest "
        "annual reports (Business, Risk Factors, MD&A, Market Risk…) with word counts and their "
        "year-on-year change; an excerpt of the chosen item; risk factors split at their headings "
        "and compared with last year: new, removed and expanded/shortened (≥ 25 %) risks."
    ),
    applies_to={"asset"},
    params=FilingsParams,
    result_kind="table",
    requires={"filings"},
)
def filings_reader(inputs: AnalysisInputs, params: FilingsParams) -> AnalysisOutput:
    data = inputs.data.get("filings")
    if not isinstance(data, dict):
        raise InsufficientDataError(
            "No filings stored: ingest them (POST /v1/filings/{symbol}/ingest)"
        )
    forms = ANNUAL if params.form == "annual" else ("10-Q",)
    docs = [d for d in data["documents"] if d["form"] in forms]
    if not docs:
        raise InsufficientDataError(f"No {params.form} document cached: ingest filings first")
    parsed = [(d, split_sections(d["text"], d["form"])) for d in docs[:2]]
    latest, sections = parsed[0]
    previous = parsed[1] if len(parsed) > 1 else None
    rows = []
    for item, sec in sections.items():
        before = previous[1].get(item) if previous else None
        rows.append(
            {
                "item": item,
                "title": sec["title"],
                "words": sec["words"],
                "previous_words": before["words"] if before else None,
                "change": (sec["words"] / before["words"] - 1)
                if before and before["words"]
                else None,
            }
        )
    chosen = sections.get(params.section)
    risks_now = risk_blocks(sections["1A"]["text"]) if "1A" in sections else []
    diff: dict[str, Any] | None = None
    if previous and "1A" in previous[1] and risks_now:
        diff = risk_diff(risk_blocks(previous[1]["1A"]["text"]), risks_now)
    return AnalysisOutput(
        "table",
        {
            "company": data.get("name"),
            "cik": data.get("cik"),
            "index_as_of": data.get("as_of"),
            "filings": [
                {k: f[k] for k in ("form", "filed", "period", "url")} for f in data["filings"][:15]
            ],
            "document": {k: latest[k] for k in ("form", "filed", "period", "url")},
            "compared_with": None
            if previous is None
            else {k: previous[0][k] for k in ("form", "filed")},
            "sections": rows,
            "section": params.section,
            "section_title": chosen["title"] if chosen else None,
            "excerpt": chosen["text"][: params.excerpt_chars] if chosen else None,
            "risk_factors": len(risks_now),
            "risk_headings": [b["heading"] for b in risks_now[:40]],
            "risk_diff": diff,
            "new_risks": len(diff["new"]) if diff else None,
            "removed_risks": len(diff["removed"]) if diff else None,
            "changed_risks": len(diff["changed"]) if diff else None,
            "new_risk_list": diff["new"] if diff else [],
            "removed_risk_list": diff["removed"] if diff else [],
            "changed_risk_list": diff["changed"] if diff else [],
            "summaries": "pending: summaries with citations need the agent runtime (J25)",
        },
    )
