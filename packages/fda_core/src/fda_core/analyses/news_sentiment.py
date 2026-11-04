"""News & sentiment (ANALYTICS_CATALOG AI-005; module M30 §4).

Input `news` = stored headlines, each with a sentiment score in [−1, 1] once rated by the
configured LLM (owner decision 2026-10-25, replaces FinBERT: no torch dependency, any provider).
Scores are third-party model output; this analysis only measures:
- headlines are assigned to the trading session on or after their UTC publication date;
- the daily index is the mean score of that session's headlines, with its headline count;
- the trailing index pools the scores of the last `window` sessions (trailing only);
- the lead–lag reading correlates the daily index with the same-session return
  (contemporaneous: news may follow the move) and with the next-session return (the news
  precedes it). Below `min_days` sessions with news the correlations are null with a reason.
Descriptive only: no signal, no verdict.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.analyses.price import _field, _prices, _windowed
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

POSITIVE, NEGATIVE = 0.15, -0.15  # label thresholds on the [−1, 1] score


class NewsParams(BaseModel):
    window: int = Field(5, ge=1, le=60, description="Trailing index window, sessions")
    min_days: int = Field(20, ge=5, le=250, description="Sessions with news for a correlation")
    headlines: int = Field(60, ge=5, le=500, description="Latest headlines listed")


def label_of(score: float | None) -> str | None:
    if score is None:
        return None
    return "positive" if score > POSITIVE else "negative" if score < NEGATIVE else "neutral"


def _corr(x: np.ndarray, y: np.ndarray, minimum: int) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < minimum:
        return {"value": None, "n": n, "reason": f"needs ≥ {minimum} sessions with news, got {n}"}
    a, b = x[ok], y[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return {"value": None, "n": n, "reason": "constant series"}
    r = float(np.corrcoef(a, b)[0, 1])
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return {"value": r, "n": n, "t_stat": t}


@analysis(
    id="AI-005",
    version="1.0.1",
    title="News & sentiment",
    description=(
        "Stored headlines with their LLM sentiment score (−1…1, labelled positive > 0.15, "
        "negative < −0.15); daily sentiment index per trading session (mean score, headline "
        "count) and a trailing index; correlation of the daily index with the same-session and "
        "next-session return. Descriptive: scores are third-party model output."
    ),
    applies_to={"asset"},
    params=NewsParams,
    result_kind="series",
    requires={"prices", "news"},
)
def news_sentiment(inputs: AnalysisInputs, params: NewsParams) -> AnalysisOutput:
    data = inputs.data.get("news")
    if not isinstance(data, dict) or not data.get("items"):
        raise InsufficientDataError(
            "No headlines stored: fetch them (POST /v1/news/{symbol}/ingest)"
        )
    frame = _windowed(inputs, _prices(inputs))
    closes = frame[_field(inputs)].astype(float)
    sessions = pd.DatetimeIndex(pd.to_datetime(closes.index)).normalize()
    items = data["items"]
    scored = [h for h in items if isinstance(h.get("sentiment"), dict)]
    by_session: dict[int, list[float]] = {}
    for h in scored:
        day = pd.Timestamp(h["published"][:10])
        at = int(sessions.searchsorted(day, side="left"))
        if 0 <= at < len(sessions) and day >= sessions[0]:
            by_session.setdefault(at, []).append(float(h["sentiment"]["score"]))
    n = len(sessions)
    daily = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for at, scores in by_session.items():
        daily[at], counts[at] = float(np.mean(scores)), len(scores)
    sums = np.array([sum(by_session.get(i, [])) for i in range(n)])
    pooled_sum = pd.Series(sums).rolling(params.window, min_periods=1).sum().to_numpy()
    pooled_n = pd.Series(counts).rolling(params.window, min_periods=1).sum().to_numpy()
    trailing = np.where(pooled_n > 0, pooled_sum / np.maximum(pooled_n, 1), np.nan)
    returns = closes.pct_change().to_numpy()
    next_returns = np.append(returns[1:], np.nan)
    labels = [label_of(float(h["sentiment"]["score"])) for h in scored]
    latest = [
        {
            "_id": h.get("id"),  # for digest sources; tables hide "_" keys
            "published": h["published"],
            "title": h["title"],
            "publisher": h.get("publisher"),
            "url": h.get("url"),
            "score": h["sentiment"]["score"] if isinstance(h.get("sentiment"), dict) else None,
            "label": label_of(h["sentiment"]["score"])
            if isinstance(h.get("sentiment"), dict)
            else None,
        }
        for h in items[: params.headlines]
    ]
    models = sorted({h["sentiment"].get("model", "") for h in scored} - {""})
    warnings = []
    if len(scored) < len(items):
        warnings.append(
            f"{len(items) - len(scored)} of {len(items)} headlines not scored yet (Score with LLM)."
        )
    warnings.append(
        "Sentiment scores are LLM output on headlines only; providers return the latest "
        "headlines, so history starts when fetching started."
    )

    def clean(values: np.ndarray) -> list[float | None]:
        return [None if not np.isfinite(v) else float(v) for v in values]

    return AnalysisOutput(
        "series",
        {
            "dates": [d.date().isoformat() for d in sessions],
            "sentiment_index": clean(trailing),
            "daily_sentiment": clean(daily),
            "headline_count": [int(c) for c in counts],
            "headlines_stored": len(items),
            "headlines_scored": len(scored),
            "sessions_with_news": int((counts > 0).sum()),
            "positive_share": labels.count("positive") / len(labels) if labels else None,
            "negative_share": labels.count("negative") / len(labels) if labels else None,
            "neutral_share": labels.count("neutral") / len(labels) if labels else None,
            "latest_index": next((v for v in clean(trailing)[::-1] if v is not None), None),
            "corr_same_session": _corr(daily, returns, params.min_days),
            "corr_next_session": _corr(daily, next_returns, params.min_days),
            "scored_by": models,
            "digest": data.get("digest"),
            "headlines": latest,
        },
        tuple(warnings),
    )
