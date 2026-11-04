"""`/v1/analyses:run` in the browser: same validation, same envelope, same problem details as
`fda_api.analyses`, but the inputs come from the exported demo bundle instead of the database
and the lake. Parameters and the range are live (the engine recomputes); inputs that the API
would resolve differently for a non-default parameter keep their exported default."""

import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from fda_contracts.envelopes import ProblemDetail, ResultEnvelope, RunLine, RunRequest
from fda_core.registry import (
    REGISTRY,
    AnalysisInputs,
    InsufficientDataError,
    NotApplicableError,
    UnknownAnalysisError,
)
from fda_demo.codec import decode

LOCKED = "/errors/demo-locked"

Loader = Callable[[str], Awaitable[Any]]
TITLES = {
    "/errors/unknown-analysis": "Unknown analysis",
    "/errors/invalid-params": "Invalid parameters",
    "/errors/not-applicable": "Not applicable",
    "/errors/insufficient-data": "Insufficient data",
    "/errors/analysis-failed": "Analysis failed",
}


def _problem(error_type: str, status: int, detail: str, analysis_id: str) -> dict[str, Any]:
    if error_type == LOCKED:
        return {
            "error": {
                "type": LOCKED,
                "title": "Full version only",
                "status": 403,
                "detail": detail,
                "analysis_id": analysis_id,
            }
        }
    line = RunLine(
        error=ProblemDetail(
            type=error_type,  # type: ignore[arg-type]
            title=TITLES[error_type],
            status=status,
            detail=detail,
            analysis_id=analysis_id,
        )
    )
    return json.loads(line.model_dump_json(exclude_none=True))


class Bundle:
    """Lazy reader of the exported inputs: one manifest per subject, one file per input blob,
    decoded once and shared by every analysis of the session."""

    def __init__(self, load: Loader) -> None:
        self._load = load
        self._manifests: dict[str, Any] = {}
        self._blobs: dict[str, Any] = {}

    async def manifest(self, subject_type: str, subject_id: str) -> Any:
        key = f"{subject_type}/{subject_id.upper() if subject_type == 'asset' else subject_id}"
        if key not in self._manifests:
            self._manifests[key] = await self._load(f"inputs/{key}.json")
        return self._manifests[key]

    async def blob(self, name: str) -> Any:
        if name not in self._blobs:
            self._blobs[name] = decode(await self._load(f"inputs/blobs/{name}.json"))
        return self._blobs[name]


async def run_one(bundle: Bundle, item: dict[str, Any], request: RunRequest) -> dict[str, Any]:
    started = time.perf_counter()
    analysis_id = str(item.get("id", ""))
    try:
        spec = REGISTRY.get(analysis_id)
    except UnknownAnalysisError as exc:
        return _problem("/errors/unknown-analysis", 404, str(exc), analysis_id)
    if request.subject.type not in spec.applies_to:
        detail = f"{spec.id} applies to {sorted(spec.applies_to)}, not {request.subject.type}"
        return _problem("/errors/not-applicable", 422, detail, spec.id)
    try:
        params = spec.params_model.model_validate(item.get("params") or {})
    except ValidationError as exc:
        return _problem("/errors/invalid-params", 422, str(exc), spec.id)

    manifest = await bundle.manifest(request.subject.type, request.subject.id)
    entry = (manifest or {}).get("analyses", {}).get(spec.id) if manifest else None
    if entry is None:
        return _problem(
            LOCKED,
            403,
            f"{request.subject.type} {request.subject.id} is not part of the online demo: run the "
            "full version (GitHub) to analyse any subject.",
            spec.id,
        )
    if "error" in entry:
        err = entry["error"]
        return _problem(err["type"], err["status"], err["detail"], spec.id)

    conventions = dict(decode(entry["conventions"])) | {"range": request.context.range}
    data = {name: await bundle.blob(blob) for name, blob in entry["inputs"].items()}
    try:
        output = spec.fn(AnalysisInputs(data=data, conventions=conventions), params)
    except InsufficientDataError as exc:
        return _problem("/errors/insufficient-data", 422, str(exc), spec.id)
    except NotApplicableError as exc:
        return _problem("/errors/not-applicable", 422, str(exc), spec.id)
    except ImportError as exc:  # libraries the browser engine does not ship (e.g. arch)
        return _problem(
            LOCKED, 403, f"Needs {exc.name or 'a native library'}: full version only.", spec.id
        )

    envelope = ResultEnvelope(
        analysis_id=spec.id,
        analysis_version=spec.version,
        subject=request.subject,
        params=params.model_dump(mode="json"),
        context=request.context,
        data_version=entry["data_version"],
        computed_at=datetime.now(UTC),
        cache="miss",
        duration_ms=(time.perf_counter() - started) * 1000,
        kind=output.kind,
        payload=dict(output.payload),
        warnings=list(output.warnings),
    )
    return json.loads(RunLine(result=envelope).model_dump_json(exclude_none=True))


async def run(bundle: Bundle, body: str) -> str:
    """NDJSON text for one request body, one line per analysis (the API's stream, in one go)."""
    request = RunRequest.model_validate_json(body)
    lines = []
    for item in json.loads(body).get("analyses", []):
        try:
            line = await run_one(bundle, item, request)
        except Exception as exc:
            line = _problem(
                "/errors/analysis-failed", 500, f"{type(exc).__name__}: {exc}", str(item.get("id"))
            )
        lines.append(json.dumps(line, allow_nan=False, default=str))
    return "\n".join(lines) + "\n"
