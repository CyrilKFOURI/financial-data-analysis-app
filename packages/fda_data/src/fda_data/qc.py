"""Price quality control (TRD §5.4)."""

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

SPIKE_THRESHOLD = 0.40


@dataclass(slots=True)
class QcReport:
    received: int = 0
    accepted: int = 0
    rejected_non_positive: int = 0
    rejected_inconsistent_range: int = 0
    rejected_duplicates: int = 0
    flagged_spikes: list[str] = field(default_factory=list)
    explained_spikes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "rejected_non_positive": self.rejected_non_positive,
            "rejected_inconsistent_range": self.rejected_inconsistent_range,
            "rejected_duplicates": self.rejected_duplicates,
            "flagged_spikes": self.flagged_spikes,
            "explained_spikes": self.explained_spikes,
        }


def check_daily_bars(
    bars: pd.DataFrame, split_dates: Collection[date] = ()
) -> tuple[pd.DataFrame, QcReport]:
    """Reject impossible bars; flag (keep) suspicious jumps unless a split explains them."""
    report = QcReport(received=len(bars))
    if bars.empty:
        return bars, report

    frame = bars.copy()
    duplicated = frame["session_date"].duplicated(keep="last")
    report.rejected_duplicates = int(duplicated.sum())
    frame = frame.loc[~duplicated]

    price_cols = ["open", "high", "low", "close", "adj_close"]
    non_positive = (frame[price_cols] <= 0).any(axis=1) | frame[price_cols].isna().any(axis=1)
    report.rejected_non_positive = int(non_positive.sum())
    frame = frame.loc[~non_positive]

    inconsistent = (frame["high"] < frame["low"]) | (frame["close"] > frame["high"] * 1.000001)
    inconsistent |= frame["close"] < frame["low"] * 0.999999
    report.rejected_inconsistent_range = int(inconsistent.sum())
    frame = frame.loc[~inconsistent].sort_values("session_date").reset_index(drop=True)

    returns = frame["adj_close"].pct_change().abs()
    splits = set(split_dates)
    for day in frame.loc[returns > SPIKE_THRESHOLD, "session_date"]:
        (report.explained_spikes if day in splits else report.flagged_spikes).append(str(day))
    report.accepted = len(frame)
    return frame, report
