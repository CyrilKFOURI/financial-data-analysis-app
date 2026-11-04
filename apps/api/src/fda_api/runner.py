"""Synchronous analysis runs outside a request (reports, alerts, snapshots): same resolver and
engine as /v1/analyses:run, without the result cache."""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy import Engine

from fda_api.resolve import Memo, resolve
from fda_contracts.envelopes import AnalysisRequestItem, RunContext, RunRequest, SubjectRef
from fda_core.registry import REGISTRY, AnalysisError
from fda_data.lake import PriceLake


@dataclass(slots=True)
class Run:
    analysis_id: str
    title: str
    version: str
    description: str
    params: dict[str, Any]
    payload: dict[str, Any] | None
    warnings: list[str]
    problem: str | None
    data_version: str | None


def run_analysis(
    engine: Engine,
    lake: PriceLake,
    subject_type: str,
    subject_id: str,
    analysis_id: str,
    params: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    memo: Memo | None = None,
) -> Run:
    spec = REGISTRY.get(analysis_id)
    params = params or {}
    base = Run(
        analysis_id, spec.title, spec.version, spec.description, params, None, [], None, None
    )
    if subject_type not in spec.applies_to:
        base.problem = f"Not applicable to a {subject_type}"
        return base
    try:
        model = spec.params_model.model_validate(params)
        request = RunRequest(
            subject=SubjectRef(type=subject_type, id=subject_id),  # type: ignore[arg-type]
            context=RunContext(**(context or {})),
            analyses=[AnalysisRequestItem(id=analysis_id, params=params)],
        )
        resolved = resolve(spec, request, engine, lake, model, {} if memo is None else memo)
        output = spec.fn(resolved.inputs(), model)
    except (AnalysisError, ValidationError, ValueError) as exc:
        base.problem = str(exc)
        return base
    base.payload = dict(output.payload)
    base.warnings = list(output.warnings)
    base.data_version = resolved.data_version
    return base


def value_at(payload: Any, path: str) -> float | None:
    """Value at a dotted path: `volatility`, `rows.0.weight`, `rows[symbol=AAPL].weight`,
    `grid.values.2.1`."""
    node = payload
    for part in path.split("."):
        if node is None:
            return None
        key, _, selector = part.partition("[")
        if key:
            if isinstance(node, list) and key.isdigit():
                index = int(key)
                node = node[index] if index < len(node) else None
            elif isinstance(node, dict):
                node = node.get(key)
            else:
                return None
        if selector:
            field, _, wanted = selector.rstrip("]").partition("=")
            node = next(
                (x for x in node or [] if isinstance(x, dict) and str(x.get(field)) == wanted), None
            )
    try:
        value = float(node)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value == value else None
