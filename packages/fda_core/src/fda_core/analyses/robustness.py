"""TR-024 Robustness protocol (Quant Rick's corrected research protocol, R-QT; feature 311).

Subject `strategy`, like the rest of the backtest lab (analyses/backtesting.py): the same
template, grid, costs, warm-up and IS/OOS split. Every choice uses in-sample data only; the OOS
Sharpe of each cell is reported for reading. V5 output: model output with its evidence.
"""

import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from fda_core import backtest as bt
from fda_core import carver
from fda_core import robustness as rb
from fda_core.analyses.backtesting import (
    LABEL,
    STRATEGY,
    _close,
    _costs,
    _grid,
    _safe,
    _split,
    _strategy,
    _target,
)
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

CARVER_RULES = tuple(carver.RULE_WEIGHTS)

# Templates that size a continuous position: the ≥ 2 trades a year rule is for on/off signals
# (a long-only Carver position rarely goes flat), so turnover is reported instead.
CONTINUOUS = {"carver_trend"}


class RobustnessParams(BaseModel):
    epoch_split: str | None = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date splitting the two epochs (default: middle of the post-warm-up sample)",
    )
    letf_baseline: Literal["off", "on"] = Field(
        "off",
        description=(
            "Add the leveraged baseline: 50/50 index buy & hold + simulated ×3 ETF above its "
            "200-day average (judged on Calmar)"
        ),
    )
    letf_fee: float = Field(0.0088, ge=0, le=0.05, description="Annual expense ratio of the ×3 ETF")
    letf_financing: float = Field(
        0.04, ge=0, le=0.2, description="Annual financing cost of the borrowed exposure"
    )


def _carver_without(
    spec: dict[str, Any], close: pd.Series, params: dict[str, float], rule: str
) -> pd.Series:
    """The Carver target without one rule (weights renormalised), overlays kept."""
    raw = carver.position(
        close,
        float(params.get("vol_target", 0.2)),
        float(params.get("buffer", 0.1)),
        not spec["short"],
        tuple(r for r in CARVER_RULES if r != rule),
    )
    return bt.apply_overlays(raw, close, spec.get("overlays") or [])


def _without_overlay(
    spec: dict[str, Any], close: pd.Series, params: dict[str, float], overlay: str
) -> pd.Series:
    kept = [o for o in spec.get("overlays") or [] if o != overlay]
    return bt.apply_overlays(
        bt.signal(spec["template"], close, params, spec["short"], spec.get("graph")), close, kept
    )


def ablation_targets(
    spec: dict[str, Any], close: pd.Series, params: dict[str, float]
) -> dict[str, pd.Series]:
    """Every removable layer switched off in turn: Carver rules and buffer, each overlay."""
    out: dict[str, pd.Series] = {}
    if spec["template"] == "carver_trend":
        for rule in CARVER_RULES:
            out[f"rule {rule}"] = _carver_without(spec, close, params, rule)
        if float(params.get("buffer", 0.0)) > 0:
            out["buffer"] = _target(spec, close, {**params, "buffer": 0.0})
    for overlay in spec.get("overlays") or []:
        out[f"overlay {overlay.replace('_', ' ')}"] = _without_overlay(spec, close, params, overlay)
    return out


def _turnover(frame: pd.DataFrame, periods: float) -> float:
    years = len(frame) / periods
    return float(frame["position"].diff().abs().sum() / years) if years > 0 else math.nan


def _calmars(rows: list[dict[str, Any]], series: str) -> list[float | None]:
    """Calmar (CAGR / |max drawdown|) of each epoch of a series."""
    out: list[float | None] = []
    for r in rows:
        if r["series"] != series:
            continue
        cagr, dd = r.get("cagr"), r.get("max_drawdown")
        out.append(cagr / -dd if cagr is not None and dd is not None and dd < 0 else None)
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.2f}"


@analysis(
    id="TR-024",
    version="1.1.1",
    title="Robustness protocol",
    description=(
        "Quant Rick's corrected protocol on the saved strategy: plateau selection on in-sample "
        "Sharpe (median of each grid cell and its neighbours, ≥ 2 trades a year, off the grid's "
        "edge); the chosen parameters traded at the decision close, one and two bars later and at "
        "the next open; two "
        "epochs fixed in advance against buy & hold, the vol-targeted buy & hold and the MA200 "
        "filter (Sharpe ± its standard error); ablation of each layer; pass/warn checks."
    ),
    applies_to=STRATEGY,
    params=RobustnessParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=90_000,
)
def robustness_protocol(inputs: AnalysisInputs, params: RobustnessParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    spec = strat["spec"]
    close = _close(prices)
    costs, warmup = _costs(spec), int(spec["warmup"])
    periods = rb.bars_per_year(close.index)
    ann = round(periods)
    grid = _grid(spec)

    # every grid cell through the engine once: in-sample Sharpe and trades a year
    frames = [bt.run_vectorised(close, _target(spec, close, g), costs, warmup) for g in grid]
    cut = _split(frames[0].index, spec["is_fraction"])
    is_parts = [f.loc[f.index < cut] for f in frames]
    is_sr = [bt.sharpe(p["net"].to_numpy(), ann) for p in is_parts]
    oos_sr = [bt.sharpe(f.loc[f.index >= cut, "net"].to_numpy(), ann) for f in frames]
    tpy = [rb.trades_per_year(p, periods) for p in is_parts]
    continuous = spec["template"] in CONTINUOUS
    plateau = rb.plateau_choice(
        grid, is_sr, tpy, spec["grid"], 0.0 if continuous else rb.MIN_TRADES_PER_YEAR
    )
    own = {k: float(v) for k, v in spec["params"].items()}
    chosen = next((i for i, g in enumerate(grid) if g == own), plateau["is_best"])
    if chosen is None:
        raise InsufficientDataError("No grid cell has a finite in-sample Sharpe")

    roles: dict[int, list[str]] = {}
    for role, i in (
        ("strategy", chosen),
        ("plateau pick", plateau["pick"]),
        ("in-sample best", plateau["is_best"]),
    ):
        if i is not None:
            roles.setdefault(i, []).append(role)
    cells = [
        {
            "role": " · ".join(names),
            **grid[i],
            "is_sharpe": _safe(is_sr[i]),
            "neighbourhood_median": _safe(plateau["scores"][i]),
            "trades_per_year": _safe(tpy[i]),
            "on_edge": plateau["edge"][i],
            "oos_sharpe_read_only": _safe(oos_sr[i]),
        }
        for i, names in roles.items()
    ]

    # execution delay
    target = _target(spec, close, grid[chosen])
    adj_open = None
    if {"open", "close"} <= set(prices.columns):
        adj_open = prices["open"].astype(float) * (
            prices["adj_close"].astype(float) / prices["close"].astype(float)
        )
    delays = rb.lag_sweep(close, target, costs, warmup, periods, adj_open)

    # two epochs fixed in advance, against the benchmarks
    sample = frames[chosen].index
    if params.epoch_split is None:
        split = pd.Timestamp(sample[len(sample) // 2])
    else:
        split = pd.Timestamp(params.epoch_split)
        if not sample[0] < split < sample[-1]:
            raise InsufficientDataError(
                f"epoch_split {params.epoch_split} is outside the sample "
                f"{sample[0].date()} → {sample[-1].date()}"
            )
    series = {"strategy": frames[chosen]}
    for name, bench in rb.benchmark_targets(
        close, float(grid[chosen].get("vol_target", 0.2))
    ).items():
        series[name] = bt.run_vectorised(close, bench, costs, warmup)
    if params.letf_baseline == "on":
        series["letf_baseline"] = rb.letf_baseline(
            close, costs, warmup, periods, 3.0, params.letf_fee, params.letf_financing
        )
    epoch_rows = rb.epochs(series, split, periods)

    # ablation
    variants = {
        layer: bt.run_vectorised(close, t, costs, warmup)
        for layer, t in ablation_targets(spec, close, grid[chosen]).items()
    }
    ablation_rows = rb.ablation(variants, frames[chosen], split, periods)

    # checks
    c_sr, c_score = is_sr[chosen], plateau["scores"][chosen]
    isolated = math.isfinite(c_sr) and c_sr > 0 and math.isfinite(c_score) and c_score < 0.5 * c_sr
    edge = plateau["edge"][chosen] and not all(plateau["edge"])
    pick = plateau["pick"]
    lag1 = next((r["sharpe"] for r in delays if r["lag"] == 1), None)
    lag2 = next((r["sharpe"] for r in delays if r["lag"] == 2), None)
    strat_e = rb.epoch_sharpes(epoch_rows, "strategy")
    hold_e = rb.epoch_sharpes(epoch_rows, "vol_targeted_hold")
    epochs_known = all(x is not None for x in (*strat_e, *hold_e))
    weak = [r["layer"] for r in ablation_rows if not r["earns_its_place"]]
    plateau_ok = not isolated and not edge
    checks = [
        rb.check(
            "Plateau",
            plateau_ok,
            "The chosen cell sits among good neighbours, off the grid's edge"
            if plateau_ok
            else ("On the grid's edge" if edge else "An isolated in-sample peak")
            + (f"; the plateau pick is {grid[pick]}" if pick is not None else ""),
        ),
        rb.check(
            "Trades a year",
            None
            if continuous or not math.isfinite(tpy[chosen])
            else tpy[chosen] >= rb.MIN_TRADES_PER_YEAR,
            f"Continuous sizing: turnover {_turnover(is_parts[chosen], periods):.1f}× a year in "
            "sample (the rule is for on/off signals)"
            if continuous
            else f"{tpy[chosen]:.1f} round trips a year in sample "
            f"(minimum {rb.MIN_TRADES_PER_YEAR:g})",
        ),
        rb.check(
            "Execution delay",
            None if lag1 is None or lag2 is None else bool(lag1 > 0 and lag2 >= 0.5 * lag1),
            f"Sharpe {_fmt(lag1)} traded at the decision close, {_fmt(lag2)} one bar later: "
            "a robust rule keeps at least half",
        ),
        rb.check(
            "Both epochs",
            all(
                s is not None and h is not None and s > h
                for s, h in zip(strat_e, hold_e, strict=True)
            )
            if epochs_known
            else None,
            "Sharpe against the vol-targeted buy & hold: "
            + ", ".join(
                f"epoch {n} {_fmt(s)} vs {_fmt(h)}"
                for n, (s, h) in enumerate(zip(strat_e, hold_e, strict=True), 1)
            ),
        ),
        rb.check(
            "Layers",
            None if not ablation_rows else not weak,
            "Nothing to remove: a single-rule template"
            if not ablation_rows
            else "Every layer lowers the Sharpe in both epochs when removed"
            if not weak
            else "Do not earn their place: " + ", ".join(weak),
        ),
    ]
    if params.letf_baseline == "on":
        mine, base = _calmars(epoch_rows, "strategy"), _calmars(epoch_rows, "letf_baseline")
        known = all(x is not None for x in (*mine, *base))
        checks.append(
            rb.check(
                "Leveraged baseline",
                all(
                    m is not None and b is not None and m > b
                    for m, b in zip(mine, base, strict=True)
                )
                if known
                else None,
                "Calmar against index buy & hold + ×3 above its 200-day average: "
                + ", ".join(
                    f"epoch {n} {_fmt(m)} vs {_fmt(b)}"
                    for n, (m, b) in enumerate(zip(mine, base, strict=True), 1)
                ),
            )
        )
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "params": grid[chosen],
            "bars_per_year": periods,
            "split_is_oos": cut.date().isoformat(),
            "epoch_split": split.date().isoformat(),
            "checks": checks,
            "plateau": cells,
            "execution": delays,
            "epochs": epoch_rows,
            "ablation": ablation_rows,
        },
    )
