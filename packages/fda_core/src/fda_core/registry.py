"""Analysis registry (TRD §3.1, §17).

Every catalog row (docs/ANALYTICS_CATALOG.md) becomes one registered analysis: a pure
function plus the metadata the API, the cache, the UI and (in V6) the agent tool layer need.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

ResultKind = Literal["value", "series", "table", "matrix", "test"]
SafetyClass = Literal["read", "write", "forbidden"]
SubjectKind = Literal[
    "asset",
    "portfolio",
    "comparison",
    "market",
    "profile",
    "derived_series",
    "universe",
    "strategy",
    "paper",
]

_ID_PATTERN = re.compile(r"^(?:[A-Z]{1,3}|DEV)-\d{3}$")
_NON_WORD = re.compile(r"[^a-z0-9]+")


class AnalysisError(Exception):
    """Base class for expected, user-facing analysis failures."""


class UnknownAnalysisError(AnalysisError):
    pass


class InsufficientDataError(AnalysisError):
    """Sample below the analysis minimum: return null + reason, never a number."""


class NotApplicableError(AnalysisError):
    """The analysis does not apply to this subject or asset type."""


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Data resolved by the API layer from `AnalysisSpec.requires`, plus explicit conventions."""

    data: Mapping[str, Any] = field(default_factory=dict)
    conventions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisOutput:
    kind: ResultKind
    payload: Mapping[str, Any]
    warnings: tuple[str, ...] = ()


type AnalysisFn[T: BaseModel] = Callable[[AnalysisInputs, T], AnalysisOutput]


@dataclass(frozen=True, slots=True)
class AnalysisSpec:
    id: str
    version: str
    title: str
    description: str
    applies_to: frozenset[SubjectKind]
    params_model: type[BaseModel]
    result_kind: ResultKind
    fn: Callable[[AnalysisInputs, Any], AnalysisOutput]
    requires: frozenset[str] = frozenset()
    asset_subclasses: frozenset[str] | None = None
    min_obs: int = 0
    budget_ms: int = 2_000
    safety_class: SafetyClass = "read"
    diagnostic: bool = False

    @property
    def tool_name(self) -> str:
        """Stable, model-facing tool name for the V6 tool layer (TRD §17)."""
        domain = self.id.split("-")[0].lower()
        slug = _NON_WORD.sub("_", self.title.lower()).strip("_")
        return f"{domain}_{slug}"

    def params_schema(self) -> dict[str, Any]:
        return self.params_model.model_json_schema()


class AnalysisRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, AnalysisSpec] = {}

    def register(self, spec: AnalysisSpec) -> None:
        if not _ID_PATTERN.match(spec.id):
            raise ValueError(f"Malformed analysis id {spec.id!r}")
        if spec.id in self._specs:
            raise ValueError(f"Analysis {spec.id} is already registered")
        if not spec.applies_to:
            raise ValueError(f"Analysis {spec.id} must apply to at least one subject kind")
        self._specs[spec.id] = spec

    def get(self, analysis_id: str) -> AnalysisSpec:
        try:
            return self._specs[analysis_id]
        except KeyError:
            raise UnknownAnalysisError(f"Unknown analysis {analysis_id!r}") from None

    def all(self) -> list[AnalysisSpec]:
        return [self._specs[key] for key in sorted(self._specs)]

    def __contains__(self, analysis_id: object) -> bool:
        return analysis_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)


REGISTRY = AnalysisRegistry()


def analysis[P: BaseModel](
    *,
    id: str,
    version: str,
    title: str,
    description: str,
    applies_to: set[SubjectKind],
    params: type[P],
    result_kind: ResultKind,
    requires: set[str] | None = None,
    asset_subclasses: set[str] | None = None,
    min_obs: int = 0,
    budget_ms: int = 2_000,
    safety_class: SafetyClass = "read",
    diagnostic: bool = False,
    registry: AnalysisRegistry = REGISTRY,
) -> Callable[[AnalysisFn[P]], AnalysisFn[P]]:
    """Register a pure analysis function under a catalog id."""

    def decorator(fn: AnalysisFn[P]) -> AnalysisFn[P]:
        registry.register(
            AnalysisSpec(
                id=id,
                version=version,
                title=title,
                description=description,
                applies_to=frozenset(applies_to),
                params_model=params,
                result_kind=result_kind,
                fn=fn,
                requires=frozenset(requires or ()),
                asset_subclasses=frozenset(asset_subclasses) if asset_subclasses else None,
                min_obs=min_obs,
                budget_ms=budget_ms,
                safety_class=safety_class,
                diagnostic=diagnostic,
            )
        )
        return fn

    return decorator
