from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import Engine, select

from fda_data.chain import NoProviderError, ProviderChain
from fda_data.ingest import default_chain, ingest_daily_prices
from fda_data.lake import PriceLake
from fda_data.providers.base import InstrumentProfile, RateLimitedError
from fda_data.providers.fake import FakeProvider
from fda_data.qc import check_daily_bars
from fda_data.ratelimit import TokenBucket
from fda_data.schema import ingestion_jobs, instruments, series_catalog

TODAY = date(2026, 10, 22)


def test_fake_provider_is_deterministic_and_refuses_real_symbols() -> None:
    provider = FakeProvider()
    first = provider.daily_bars("FAKE", date(2024, 1, 1), date(2024, 3, 1))
    second = provider.daily_bars("fake", date(2024, 1, 1), date(2024, 3, 1))
    pd.testing.assert_frame_equal(first, second)
    assert (first["high"] >= first[["open", "close"]].max(axis=1)).all()
    assert not provider.supports("AAPL")


def test_lake_round_trip_and_merge(lake: PriceLake) -> None:
    bars = (
        FakeProvider()
        .daily_bars("FAKE", date(2023, 12, 1), date(2024, 1, 31))
        .assign(source="fake")
    )
    lake.write_prices("equity", 1, bars.iloc[:20])
    lake.write_prices(
        "equity", 1, bars
    )  # overlapping write: same dates are replaced, not duplicated

    out = lake.read_prices("equity", 1)
    assert len(out) == len(bars)
    assert out["session_date"].is_monotonic_increasing
    january = lake.read_prices("equity", 1, start=date(2024, 1, 1), end=date(2024, 1, 31))
    assert pd.to_datetime(january["session_date"]).dt.month.eq(1).all()


def test_qc_rejects_impossible_bars_and_flags_spikes() -> None:
    bars = pd.DataFrame(
        {
            "session_date": [date(2024, 1, d) for d in (2, 3, 4, 5, 5)],
            "open": [10.0, 10.0, -1.0, 10.0, 15.0],
            "high": [11.0, 9.0, 11.0, 16.0, 16.0],
            "low": [9.0, 10.0, 9.0, 9.0, 14.0],
            "close": [10.0, 9.5, 10.0, 15.0, 15.0],
            "adj_close": [10.0, 9.5, 10.0, 15.0, 15.0],
            "volume": [1.0] * 5,
        }
    )
    clean, report = check_daily_bars(bars)

    assert report.rejected_duplicates == 1
    assert report.rejected_non_positive == 1
    assert report.rejected_inconsistent_range == 1  # high < low on Jan 3
    assert report.accepted == 2
    assert report.flagged_spikes == ["2024-01-05"]  # +50 % without corporate action
    assert list(clean["session_date"]) == [date(2024, 1, 2), date(2024, 1, 5)]


class ThrottledProvider(FakeProvider):
    name = "throttled"

    def profile(self, symbol: str) -> InstrumentProfile:
        raise RateLimitedError("429")


def test_chain_falls_back_on_rate_limit_and_records_source() -> None:
    chain = ProviderChain([ThrottledProvider(), FakeProvider()])
    profile = chain.profile("FAKE")
    assert profile.source == "fake"


def test_chain_reports_when_nobody_serves_a_symbol() -> None:
    with pytest.raises(NoProviderError, match="No provider can serve AAPL"):
        default_chain().profile("AAPL")


def test_token_bucket_limits_bursts() -> None:
    bucket = TokenBucket(rate_per_second=1, burst=2)
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()


def test_ingestion_is_incremental_and_tracked(engine: Engine, lake: PriceLake) -> None:
    first = ingest_daily_prices(engine, lake, default_chain(), "FAKE", today=TODAY)
    second = ingest_daily_prices(engine, lake, default_chain(), "FAKE", today=TODAY)

    assert first.status == "succeeded"
    assert 2_500 <= first.rows_written <= 2_620  # ~10 years of business days
    assert second.status == "succeeded"
    assert second.rows_written == 0
    with engine.connect() as conn:
        catalog = conn.execute(select(series_catalog)).mappings().one()
        jobs = conn.execute(select(ingestion_jobs.c.status)).scalars().all()
        instrument = conn.execute(select(instruments)).mappings().one()
    assert catalog["rows"] == first.rows_written
    assert catalog["source"] == "fake"
    assert catalog["last_date"] <= TODAY
    assert jobs == ["succeeded", "succeeded"]
    assert instrument["symbol"] == "FAKE"
    assert instrument["asset_class"] == "equity"
    assert len(lake.read_prices("equity", instrument["id"])) == first.rows_written


def test_ingesting_a_real_symbol_fails_cleanly_without_fake_data(
    engine: Engine, lake: PriceLake
) -> None:
    result = ingest_daily_prices(engine, lake, default_chain(), "AAPL", today=TODAY)

    assert result.status == "failed"
    assert "No provider can serve AAPL" in result.message
    with engine.connect() as conn:
        assert conn.execute(select(instruments)).first() is None


def test_refresh_overlaps_recent_days_and_never_stores_today(
    engine: Engine, lake: PriceLake
) -> None:
    from datetime import timedelta

    first = ingest_daily_prices(engine, lake, default_chain(), "FAKE", today=TODAY)
    later = ingest_daily_prices(
        engine, lake, default_chain(), "FAKE", today=TODAY + timedelta(days=7)
    )
    stored = lake.read_prices("equity", first.instrument_id or 0)
    with engine.connect() as conn:
        catalog = conn.execute(select(series_catalog)).mappings().one()

    assert later.rows_written > 5  # the overlap is re-read, then the new sessions
    assert catalog["rows"] == len(stored)  # coverage is measured, not accumulated
    last_stored = pd.Timestamp(stored["session_date"].max()).date()
    assert last_stored < TODAY + timedelta(days=7)
    assert catalog["last_date"] == last_stored


def test_long_history_symbols_are_backfilled_once() -> None:
    from fda_data.ingest import price_window

    today = date(2026, 10, 24)
    short = {"first_date": date(2016, 10, 25), "last_date": date(2026, 10, 22)}
    start, end = price_window("^GSPC", short, today) or (None, None)
    assert start == today - timedelta(days=365 * 60)
    assert end == date(2026, 10, 23)
    full = {"first_date": date(1966, 11, 1), "last_date": date(2026, 10, 22)}
    assert price_window("^GSPC", full, today) == (date(2026, 10, 17), date(2026, 10, 23))
    assert price_window("AAPL", short, today) == (date(2026, 10, 17), date(2026, 10, 23))
    assert price_window("^GSPC", None, today) == (
        today - timedelta(days=365 * 60),
        date(2026, 10, 23),
    )
