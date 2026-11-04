"""Golden test (hand-computed values) for AI-005: session assignment, daily and trailing index,
label shares, null correlations below the minimum sample."""

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.news_sentiment import NewsParams, label_of, news_sentiment
from fda_core.registry import AnalysisInputs, InsufficientDataError

CONV = {"annualisation": 252, "price_field": "adj_close", "range": "1Y"}


def _inputs(items: list[dict[str, object]]) -> AnalysisInputs:
    close = np.array([100.0, 102.0, 101.0, 103.0, 104.0])  # Tue 2 → Mon 8 Jan 2024
    index = pd.date_range("2024-01-02", periods=5, freq="B")
    prices = pd.DataFrame({"close": close, "adj_close": close}, index=index)
    return AnalysisInputs(data={"prices": prices, "news": {"items": items}}, conventions=CONV)


def _h(published: str, score: float | None) -> dict[str, object]:
    sentiment = None if score is None else {"score": score, "model": "gemini:g"}
    return {"id": published, "published": published, "title": "t", "sentiment": sentiment}


def test_golden_index() -> None:
    items = [
        _h("2024-01-06T09:00:00+00:00", 0.8),  # Saturday → Monday 8 Jan (session 4)
        _h("2024-01-04T12:00:00+00:00", None),  # not scored: ignored
        _h("2024-01-02T15:00:00+00:00", -0.1),
        _h("2024-01-02T10:00:00+00:00", 0.5),
        _h("2024-01-01T10:00:00+00:00", 1.0),  # before the first session: dropped
    ]
    out = news_sentiment(_inputs(items), NewsParams(window=2, min_days=5)).payload
    assert out["dates"][0] == "2024-01-02"
    assert out["headline_count"] == [2, 0, 0, 0, 1]
    assert out["daily_sentiment"] == [pytest.approx(0.2), None, None, None, pytest.approx(0.8)]
    # pooled over 2 sessions: (0.5 − 0.1) / 2, same, no news, no news, 0.8 / 1
    assert out["sentiment_index"] == [
        pytest.approx(0.2),
        pytest.approx(0.2),
        None,
        None,
        pytest.approx(0.8),
    ]
    assert out["latest_index"] == pytest.approx(0.8)
    assert out["headlines_stored"] == 5
    assert out["headlines_scored"] == 4
    assert out["sessions_with_news"] == 2
    assert out["positive_share"] == 0.75  # 0.8, 0.5, 1.0 positive; −0.1 neutral
    assert out["neutral_share"] == 0.25
    assert out["corr_same_session"]["value"] is None
    assert out["corr_next_session"]["reason"].startswith("needs ≥ 5 sessions")
    assert out["scored_by"] == ["gemini:g"]
    assert out["headlines"][1]["label"] is None


def test_labels_and_missing_news() -> None:
    assert [label_of(x) for x in (0.16, 0.15, -0.15, -0.16, None)] == [
        "positive",
        "neutral",
        "neutral",
        "negative",
        None,
    ]
    with pytest.raises(InsufficientDataError):
        news_sentiment(_inputs([]), NewsParams())
