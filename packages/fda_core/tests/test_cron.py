from datetime import UTC, datetime

import pytest

from fda_core.cron import CronError, parse

T = datetime(2026, 10, 24, 10, 7, tzinfo=UTC)  # Saturday


def test_next_runs() -> None:
    assert parse("*/15 * * * *").next_after(T) == T.replace(minute=15)
    assert parse("30 18 * * 1-5").next_after(T) == datetime(
        2026, 10, 26, 18, 30, tzinfo=UTC
    )  # Monday
    assert parse("0 7 1 * *").next_after(T) == datetime(2026, 11, 1, 7, 0, tzinfo=UTC)
    assert parse("0 9 * * 0").next_after(T) == datetime(2026, 10, 25, 9, 0, tzinfo=UTC)  # Sunday
    assert parse("0 9 * * 7").next_after(T) == datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
    assert parse("0 0 29 2 *").next_after(T) == datetime(2028, 2, 29, 0, 0, tzinfo=UTC)
    # both day fields restricted: either matches (the 1st, or any Monday)
    assert parse("0 6 1 * 1").next_after(T) == datetime(2026, 10, 26, 6, 0, tzinfo=UTC)
    assert parse("7 10 * * *").next_after(T) == datetime(
        2026, 10, 25, 10, 7, tzinfo=UTC
    )  # strictly after


@pytest.mark.parametrize(
    "bad", ["* * * *", "60 * * * *", "* 24 * * *", "*/0 * * * *", "a * * * *", "5-1 * * * *"]
)
def test_invalid(bad: str) -> None:
    with pytest.raises(CronError):
        parse(bad)
