import asyncio

import pytest
from arq import Retry

from fda_data.ingest import IngestResult
from fda_worker.settings import RETRY_BASE_S, WorkerSettings, _outcome, ping


def test_ping_task_is_registered_and_answers() -> None:
    assert ping in WorkerSettings.functions
    assert asyncio.run(ping({})) == "pong"


def test_throttled_jobs_are_retried_with_backoff_then_reported() -> None:
    queued = IngestResult(job_id=1, symbol="AAPL", status="queued", retryable=True)
    with pytest.raises(Retry) as retry:
        _outcome({"job_try": 2}, queued)
    assert retry.value.defer_score == RETRY_BASE_S * 2 * 1000

    failed = IngestResult(job_id=1, symbol="AAPL", status="failed", retryable=True)
    assert _outcome({"job_try": 3}, failed)["status"] == "failed"
