"""TR-025 Signal ensembles & shadow test (Quant Rick bootcamp 1: ensembles, Boruta; feature 311).

Subject `strategy`. Members = the k configurations of the strategy's grid with the best
in-sample Sharpe (chosen on in-sample data only). They are combined by OR, AND, majority vote
and mean and run through the same engine; every member and ensemble faces its shadows, the same
position path rotated in time (a Boruta-style permutation test on in-sample bars).
"""

import math
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from fda_core import backtest as bt
from fda_core import robustness as rb
from fda_core.analyses.backtesting import (
    LABEL,
    STRATEGY,
    _close,
    _costs,
    _grid,
    _split,
    _strategy,
    _target,
)
from fda_core.registry import AnalysisInputs, AnalysisOutput, analysis

SIGNIFICANCE = 0.05


class EnsembleParams(BaseModel):
    members: int = Field(5, ge=2, le=12, description="Best in-sample configurations combined")
    shadows: int = Field(200, ge=50, le=1000, description="Rotated shadows per series")


def _row(
    name: str,
    frame: pd.DataFrame,
    cut: pd.Timestamp,
    costs: bt.Costs,
    periods: float,
    shadows: int,
) -> dict[str, Any]:
    is_part, oos_part = frame.loc[frame.index < cut], frame.loc[frame.index >= cut]
    shadow = rb.shadow_test(is_part, costs, periods, shadows)
    oos = rb.summary(oos_part["net"], periods)
    p = shadow["p_value"]
    return {
        "series": name,
        "is_sharpe": shadow["sharpe"],
        "shadow_p95": shadow["shadow_p95"],
        "p_value": p,
        "beats_its_shadows": p is not None and p <= SIGNIFICANCE,
        "oos_sharpe_read_only": oos["sharpe"],
        "oos_max_drawdown": oos["max_drawdown"],
        "exposure": float(frame["position"].abs().mean()),
    }


@analysis(
    id="TR-025",
    version="1.0.0",
    title="Signal ensembles & shadow test",
    description=(
        "The k best in-sample configurations of the strategy's grid combined by OR, AND, majority "
        "vote and mean; every member and ensemble against its shadows (its own position path "
        "rotated in time: same exposure, timing destroyed), p = (1 + #shadows ≥ actual)/(n + 1) "
        "on in-sample bars; OOS Sharpe and drawdown shown read-only."
    ),
    applies_to=STRATEGY,
    params=EnsembleParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=90_000,
)
def signal_ensembles(inputs: AnalysisInputs, params: EnsembleParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    spec = strat["spec"]
    close = _close(prices)
    costs, warmup = _costs(spec), int(spec["warmup"])
    periods = rb.bars_per_year(close.index)
    grid = _grid(spec)
    targets = [_target(spec, close, g) for g in grid]
    frames = [bt.run_vectorised(close, t, costs, warmup) for t in targets]
    cut = _split(frames[0].index, spec["is_fraction"])
    is_sr = [bt.sharpe(f.loc[f.index < cut, "net"].to_numpy(), round(periods)) for f in frames]
    ranked = sorted(
        (i for i in range(len(grid)) if math.isfinite(is_sr[i])), key=lambda i: -is_sr[i]
    )
    ranked = ranked[: params.members]
    members = pd.DataFrame({i: targets[i] for i in ranked})
    rows = [
        _row(
            "member " + ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in grid[i].items()),
            frames[i],
            cut,
            costs,
            periods,
            params.shadows,
        )
        for i in ranked
    ]
    if len(ranked) >= 2:
        for rule in rb.ENSEMBLE_RULES:
            frame = bt.run_vectorised(close, rb.combine(members, rule), costs, warmup)
            rows.append(_row(f"ensemble {rule}", frame, cut, costs, periods, params.shadows))
    beating = sum(bool(r["beats_its_shadows"]) for r in rows)
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "members": len(ranked),
            "split_is_oos": cut.date().isoformat(),
            "rows": rows,
            "note": (
                "Members are chosen on in-sample Sharpe only and the shadow test uses in-sample "
                f"bars. {beating} of {len(rows)} series beat their shadows at 5 %."
            ),
        },
    )
