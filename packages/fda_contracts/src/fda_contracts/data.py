"""Wire contracts of the data endpoints (M02)."""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class InstrumentSummary(BaseModel):
    symbol: str
    name: str
    asset_class: str
    asset_subclass: str | None
    currency: str
    ingested: bool
    source: str | None = None
    profile: dict[str, Any] = {}


class PriceSeries(BaseModel):
    """Columnar daily bars (compact on the wire)."""

    symbol: str
    currency: str
    source: str | None
    dates: list[date]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    adj_close: list[float]
    volume: list[float]


class CorporateAction(BaseModel):
    ex_date: date
    action: Literal["dividend", "split", "spinoff"]
    ratio: float | None
    amount: float | None
    currency: str | None
    source: str | None


class SeriesData(BaseModel):
    """Levels by date (rates in percent, FX in quote per base): not tradable prices."""

    code: str
    title: str
    unit: str
    frequency: str
    source: str | None
    dates: list[date]
    values: list[float]


class DatasetStatus(BaseModel):
    dataset: str
    first_date: date | None
    last_date: date | None
    rows: int
    source: str | None
    last_ingested_at: datetime | None


class JobInfo(BaseModel):
    id: int
    kind: str
    status: JobStatus
    progress: float
    message: str | None
    provider: str | None
    rows_written: int
    qc_summary: dict[str, Any] | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class DataStatus(BaseModel):
    symbol: str
    instrument: InstrumentSummary | None
    datasets: list[DatasetStatus]
    jobs: list[JobInfo]
    can_ingest: bool
    reason: str | None


class JobAccepted(BaseModel):
    job_id: int
    runner: Literal["worker", "inline"]


class SeriesStatus(BaseModel):
    code: str
    title: str | None
    unit: str | None
    provider: str | None
    dataset: DatasetStatus | None
    jobs: list[JobInfo]
    can_ingest: bool
    reason: str | None


class TypeConventions(BaseModel):
    annualisation: int | None
    calendar: str
    price_field: str
    has_volume: bool
    return_kind: str
    benchmark: str | None
    caveats: list[str]


class AssetType(BaseModel):
    """Refined classification (ASSET_TYPES §2) and the conventions analyses inherit (§3)."""

    symbol: str
    asset_class: str
    asset_subclass: str | None
    leverage_factor: float | None
    confidence: str | None
    source: str | None
    reason: str | None
    flags: list[str]
    overrides: dict[str, str]
    conventions: TypeConventions


class OverrideRequest(BaseModel):
    field: Literal["asset_class", "asset_subclass", "leverage_factor"]
    value: str = Field(min_length=1, max_length=32)
    reason: str | None = Field(default=None, max_length=200)
