"""J26 (AI-006) guardrails on the investment (IN-) and trading (TR-) tools.

V4–V5 results are model outputs (CLAUDE.md domain rules). Every IN-/TR- tool result is therefore
labelled as a model output and carries its assumptions. A result that states performance (Sharpe,
CAGR, returns…) is only returned with out-of-sample evidence:
- the evidence already in the payload (DSR, PBO, OOS, walk-forward, placebo, naive baselines);
- otherwise the validation analysis of the subject, run by the tool layer: TR-007 overfitting
  control for a strategy, IN-018 walk-forward for an optimiser on a portfolio.

When that validation cannot be computed, the performance figures are withheld rather than
returned bare. Paper-trading records (TR-040, TR-042) are simulated fills, not backtests, and
are exempt. There is no live-order tool: orders exist only as paper write actions approved in
the web app.
"""

import re
from typing import Any

MODEL_FAMILIES = ("IN-", "TR-")
LABEL = (
    "Model output: assumptions and out-of-sample evidence are attached; not a recommendation "
    "and not a forecast of future results."
)
PERFORMANCE = re.compile(
    r"sharpe|sortino|calmar|cagr|total_return|expected_return|annual_return|final_equity|"
    r"^performance$|^equity$"
)
EVIDENCE = re.compile(
    r"dsr|pbo|deflated|psr|haircut|oos|out_of_sample|walk_forward|walk_backward|validation|"
    r"placebo|naive|baseline|beats_|trials|historical_evidence|monte_carlo|degradation"
)
EVIDENCE_ANALYSIS = {"strategy": "TR-007", "portfolio": "IN-018"}
EXEMPT = frozenset({"TR-040", "TR-042"})  # simulated fills reconcile to the cent: records


def is_model_output(analysis_id: str | None) -> bool:
    return bool(analysis_id) and str(analysis_id).startswith(MODEL_FAMILIES)


def performance_keys(payload: dict[str, Any]) -> list[str]:
    return sorted(k for k in payload if PERFORMANCE.search(k) and not EVIDENCE.search(k))


def evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload's own out-of-sample evidence (a bare `validated: false` is not evidence)."""
    found = {k: v for k, v in payload.items() if EVIDENCE.search(k)}
    if payload.get("validated") is True:
        found["validated"] = True
    return found


def needs_validation(analysis_id: str, payload: dict[str, Any]) -> bool:
    if analysis_id in EXEMPT or not performance_keys(payload):
        return False
    return not evidence(payload)


def evidence_request(
    subject_type: str, payload: dict[str, Any], params: dict[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    """Which validation analysis backs this result, with the parameters it shares."""
    analysis_id = EVIDENCE_ANALYSIS.get(subject_type)
    if analysis_id is None:
        return None
    if analysis_id == "IN-018":
        shared = {k: v for k, v in params.items() if k not in ("compare", "trials")}
        if payload.get("method"):
            shared["method"] = payload["method"]
        return analysis_id, shared
    return analysis_id, {}


def withhold(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload without its performance figures (validation unavailable)."""
    hidden = set(performance_keys(payload))
    return {k: v for k, v in payload.items() if k not in hidden}
