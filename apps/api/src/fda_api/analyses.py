"""Analysis endpoints: catalogue listing and streamed execution (TRD §7.1, §8)."""

import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

import fda_core.analyses  # noqa: F401  (registers analyses)
from fda_api.cache import ResultCache, cache_key
from fda_api.resolve import Memo, resolve
from fda_contracts.envelopes import (
    AnalysisInfo,
    AnalysisRequestItem,
    ErrorType,
    ProblemDetail,
    ResultEnvelope,
    RunLine,
    RunRequest,
)
from fda_core.registry import (
    REGISTRY,
    AnalysisSpec,
    InsufficientDataError,
    NotApplicableError,
    UnknownAnalysisError,
)

router = APIRouter(prefix="/v1", tags=["analyses"])


def _info(spec: AnalysisSpec) -> AnalysisInfo:
    return AnalysisInfo(
        id=spec.id,
        version=spec.version,
        title=spec.title,
        description=spec.description,
        applies_to=sorted(spec.applies_to),
        asset_subclasses=sorted(spec.asset_subclasses) if spec.asset_subclasses else None,
        requires=sorted(spec.requires),
        result_kind=spec.result_kind,
        min_obs=spec.min_obs,
        budget_ms=spec.budget_ms,
        safety_class=spec.safety_class,
        diagnostic=spec.diagnostic,
        tool_name=spec.tool_name,
        params_schema=spec.params_schema(),
    )


@router.get("/analyses")
def list_analyses() -> list[AnalysisInfo]:
    return [_info(spec) for spec in REGISTRY.all()]


def _problem(error_type: ErrorType, status: int, detail: str, analysis_id: str) -> RunLine:
    titles: dict[ErrorType, str] = {
        "/errors/unknown-analysis": "Unknown analysis",
        "/errors/invalid-params": "Invalid parameters",
        "/errors/not-applicable": "Not applicable",
        "/errors/insufficient-data": "Insufficient data",
        "/errors/analysis-failed": "Analysis failed",
    }
    return RunLine(
        error=ProblemDetail(
            type=error_type,
            title=titles[error_type],
            status=status,
            detail=detail,
            analysis_id=analysis_id,
        )
    )


async def _run_one(
    item: AnalysisRequestItem, request: RunRequest, http_request: Request, memo: Memo
) -> RunLine:
    cache: ResultCache = http_request.app.state.cache
    started = time.perf_counter()
    try:
        spec = REGISTRY.get(item.id)
    except UnknownAnalysisError as exc:
        return _problem("/errors/unknown-analysis", 404, str(exc), item.id)

    if request.subject.type not in spec.applies_to:
        detail = f"{spec.id} applies to {sorted(spec.applies_to)}, not {request.subject.type}"
        return _problem("/errors/not-applicable", 422, detail, spec.id)

    try:
        params = spec.params_model.model_validate(item.params)
    except ValidationError as exc:
        return _problem("/errors/invalid-params", 422, str(exc), spec.id)

    canonical_params = params.model_dump(mode="json")
    try:
        resolved = await anyio.to_thread.run_sync(
            resolve,
            spec,
            request,
            http_request.app.state.engine,
            http_request.app.state.lake,
            params,
            memo,
        )
    except InsufficientDataError as exc:
        return _problem("/errors/insufficient-data", 422, str(exc), spec.id)
    key = cache_key(
        analysis_id=spec.id,
        analysis_version=spec.version,
        params=canonical_params,
        subject=request.subject.model_dump(),
        context=request.context.model_dump(),
        data_version=resolved.data_version,
    )

    cached = await cache.get(key)
    if cached is not None:
        envelope = ResultEnvelope.model_validate_json(cached)
        envelope.cache = "hit"
        envelope.duration_ms = (time.perf_counter() - started) * 1000
        return RunLine(result=envelope)

    def compute() -> Any:
        return spec.fn(resolved.inputs(), params)  # frames are loaded on a cache miss only

    try:
        output = await anyio.to_thread.run_sync(compute)
    except InsufficientDataError as exc:
        return _problem("/errors/insufficient-data", 422, str(exc), spec.id)
    except NotApplicableError as exc:
        return _problem("/errors/not-applicable", 422, str(exc), spec.id)

    envelope = ResultEnvelope(
        analysis_id=spec.id,
        analysis_version=spec.version,
        subject=request.subject,
        params=canonical_params,
        context=request.context,
        data_version=resolved.data_version,
        computed_at=datetime.now(UTC),
        cache="miss",
        duration_ms=(time.perf_counter() - started) * 1000,
        kind=output.kind,
        payload=dict(output.payload),
        warnings=list(output.warnings),
    )
    await cache.set(key, envelope.model_dump_json().encode())
    return RunLine(result=envelope)


@router.post(
    "/analyses:run",
    response_model=RunLine,
    responses={
        200: {"content": {"application/x-ndjson": {}}, "description": "One RunLine per line"}
    },
)
async def run_analyses(request: RunRequest, http_request: Request) -> StreamingResponse:
    async def lines() -> AsyncIterator[str]:
        memo: Memo = {}  # catalog rows and frames shared by the analyses of this request
        for item in request.analyses:
            line = await _run_one(item, request, http_request, memo)
            yield line.model_dump_json(exclude_none=True) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
