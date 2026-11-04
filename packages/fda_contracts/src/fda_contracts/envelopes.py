"""Wire contracts of the analysis API (TRD §3.1, §8)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ErrorType = Literal[
    "/errors/unknown-analysis",
    "/errors/invalid-params",
    "/errors/not-applicable",
    "/errors/insufficient-data",
    "/errors/analysis-failed",
]


class SubjectRef(BaseModel):
    type: Literal[
        "asset", "portfolio", "comparison", "market", "universe", "profile", "strategy", "paper"
    ]
    id: str = Field(min_length=1, max_length=64, description="Symbol or object id")


class RunContext(BaseModel):
    range: str = Field("1Y", max_length=24, pattern=r"^[0-9A-Z.\-]+$")
    freq: Literal["D", "W", "M"] = "D"
    ccy: str = Field("listing", max_length=8, pattern=r"^[A-Za-z]+$")
    bench: str = Field("auto", max_length=20, pattern=r"^[\w.^=-]+$")


class AnalysisRequestItem(BaseModel):
    id: str = Field(max_length=16)
    params: dict[str, Any] = Field(default_factory=dict, max_length=32)


class RunRequest(BaseModel):
    subject: SubjectRef
    context: RunContext = Field(default_factory=RunContext)
    analyses: list[AnalysisRequestItem] = Field(min_length=1, max_length=50)


class ResultEnvelope(BaseModel):
    analysis_id: str
    analysis_version: str
    subject: SubjectRef
    params: dict[str, Any]
    context: RunContext
    data_version: str
    computed_at: datetime
    cache: Literal["hit", "miss"]
    duration_ms: float
    kind: Literal["value", "series", "table", "matrix", "test"]
    payload: dict[str, Any]
    warnings: list[str]


class ProblemDetail(BaseModel):
    """RFC 9457 problem details, one per failed analysis line."""

    type: ErrorType
    title: str
    status: int
    detail: str
    analysis_id: str | None = None


class RunLine(BaseModel):
    """One NDJSON line of `POST /v1/analyses:run`: exactly one of result / error is set."""

    result: ResultEnvelope | None = None
    error: ProblemDetail | None = None


class AnalysisInfo(BaseModel):
    id: str
    version: str
    title: str
    description: str
    applies_to: list[str]
    asset_subclasses: list[str] | None
    requires: list[str]
    result_kind: str
    min_obs: int
    budget_ms: int
    safety_class: str
    diagnostic: bool
    tool_name: str
    params_schema: dict[str, Any]
