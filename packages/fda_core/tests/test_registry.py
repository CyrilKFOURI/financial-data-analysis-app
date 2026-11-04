import pytest
from pydantic import BaseModel

import fda_core.analyses  # noqa: F401  (registers analyses)
from fda_core.analyses.diagnostics import SyntheticStatisticsParams, synthetic_statistics
from fda_core.registry import (
    REGISTRY,
    AnalysisInputs,
    AnalysisOutput,
    AnalysisRegistry,
    UnknownAnalysisError,
    analysis,
)


class NoParams(BaseModel):
    pass


def _register(registry: AnalysisRegistry, analysis_id: str) -> None:
    @analysis(
        id=analysis_id,
        version="1.0.0",
        title="Test analysis",
        description="test",
        applies_to={"asset"},
        params=NoParams,
        result_kind="value",
        registry=registry,
    )
    def fn(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
        return AnalysisOutput(kind="value", payload={"value": 1})


def test_duplicate_ids_are_rejected() -> None:
    registry = AnalysisRegistry()
    _register(registry, "T-999")
    with pytest.raises(ValueError, match="already registered"):
        _register(registry, "T-999")


@pytest.mark.parametrize("bad_id", ["T999", "t-001", "TOOLONG-001", "T-01", "R-0001"])
def test_malformed_ids_are_rejected(bad_id: str) -> None:
    with pytest.raises(ValueError, match="Malformed"):
        _register(AnalysisRegistry(), bad_id)


def test_unknown_id_raises_domain_error() -> None:
    with pytest.raises(UnknownAnalysisError):
        AnalysisRegistry().get("T-001")


def test_tool_metadata_is_derived_from_catalog_fields() -> None:
    spec = REGISTRY.get("DEV-001")
    assert spec.tool_name == "dev_synthetic_statistics"
    assert spec.params_schema()["properties"]["sigma"]["exclusiveMinimum"] == 0.0
    assert spec.diagnostic


def test_synthetic_statistics_is_deterministic_and_recovers_parameters() -> None:
    params = SyntheticStatisticsParams(n=100_000, seed=1, mu=0.08, sigma=0.20)
    first = synthetic_statistics(AnalysisInputs(), params)
    second = synthetic_statistics(AnalysisInputs(), params)
    assert first == second
    assert first.payload["vol_annualised"] == pytest.approx(0.20, abs=0.002)
    assert first.payload["mean_annualised"] == pytest.approx(0.08, abs=0.03)
