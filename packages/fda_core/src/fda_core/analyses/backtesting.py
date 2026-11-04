"""Backtest lab (ANALYTICS_CATALOG TR-001…TR-009, TR-014…TR-017; module M25).

Subject `strategy`: `strategy` = {id, name, hypothesis, spec, trials}, `prices` = the symbol's
daily bars. Protocol (TEMA notebook, M25 §1): warm-up 220 bars, IS/OOS 60/40 by time, signals
shifted one bar, 5 bp fees + 5 bp slippage by default, grid search over the spec's grid, every
grid configuration counted as a trial (ledger total feeds DSR). Results are only returned
together with their validation (DSR, PBO, look-ahead check); V5 outputs are model outputs.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core import backtest as bt
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

STRATEGY: set[Any] = {"strategy"}
LABEL = "Model output — backtest evidence, not a recommendation"


def _strategy(inputs: AnalysisInputs) -> tuple[dict[str, Any], pd.DataFrame]:
    strat = inputs.data.get("strategy")
    prices = inputs.data.get("prices")
    if not isinstance(strat, dict):
        raise InsufficientDataError("Strategy not loaded")
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise InsufficientDataError(
            f"No stored prices for {strat['spec']['symbol']}: ingest them first"
        )
    spec = strat["spec"]
    if len(prices) < spec["warmup"] + 252:
        raise InsufficientDataError(
            f"Needs ≥ {spec['warmup'] + 252} bars (warm-up + 1 year), got {len(prices)}"
        )
    return strat, prices


def _target(spec: dict[str, Any], close: pd.Series, params: dict[str, float]) -> pd.Series:
    signal = bt.signal(spec["template"], close, params, spec["short"])
    return bt.apply_overlays(signal, close, spec.get("overlays") or [])


def _close(prices: pd.DataFrame) -> pd.Series:
    return prices["adj_close"].astype(float)


def _costs(spec: dict[str, Any]) -> bt.Costs:
    return bt.Costs(
        float(spec["costs"].get("fee_bps", 5)), float(spec["costs"].get("slippage_bps", 5))
    )


def _grid(spec: dict[str, Any]) -> list[dict[str, float]]:
    grid = bt.param_grid(
        {k: [float(x) for x in v] for k, v in spec["grid"].items()},
        bt.ordered_constraint(spec["template"]),
    )
    if not grid:
        raise InsufficientDataError("The parameter grid is empty after constraints")
    return grid


def _split(index: pd.Index, fraction: float) -> pd.Timestamp:
    return pd.Timestamp(index[int(len(index) * fraction)])


def _evaluate(spec: dict[str, Any], close: pd.Series) -> dict[str, Any]:
    """Shared core: grid trial matrix on the full period, IS/OOS split, chosen params (spec
    params, or the IS-best when absent), DSR with the ledger's trial count, PBO, look-ahead."""
    costs, warmup = _costs(spec), int(spec["warmup"])
    grid = _grid(spec)
    trials = bt.grid_returns(
        spec["template"], close, grid, costs, warmup, spec["short"], spec.get("overlays")
    )
    cut = _split(trials.index, spec["is_fraction"])
    is_part, oos_part = trials.loc[trials.index < cut], trials.loc[trials.index >= cut]
    is_sr = is_part.apply(lambda c: bt.sharpe(c.to_numpy()))
    oos_sr = oos_part.apply(lambda c: bt.sharpe(c.to_numpy()))
    params = {k: float(v) for k, v in spec["params"].items()}
    chosen = next((i for i, g in enumerate(grid) if g == params), int(is_sr.idxmax()))
    return {
        "grid": grid,
        "trials": trials,
        "cut": cut,
        "is_sr": is_sr,
        "oos_sr": oos_sr,
        "chosen": chosen,
        "params": grid[chosen],
        "costs": costs,
        "warmup": warmup,
    }


def _validation(strat: dict[str, Any], ev: dict[str, Any], close: pd.Series) -> dict[str, Any]:
    trials: pd.DataFrame = ev["trials"]
    oos = trials.loc[trials.index >= ev["cut"], ev["chosen"]].to_numpy()
    per_period = [
        float(c.mean() / c.std(ddof=1))
        for _, c in trials.loc[trials.index >= ev["cut"]].items()
        if c.std(ddof=1) > 0
    ]
    ledger = max(int(strat.get("trials") or 0), len(ev["grid"]))
    srs = per_period + [0.0] * max(0, ledger - len(per_period))
    pbo = bt.pbo(trials, 16) if trials.shape[1] >= 2 and len(trials) >= 320 else None
    target = _target(strat["spec"], close, ev["params"])
    look = bt.lookahead_check(close, target, ev["costs"], ev["warmup"])
    return {
        "trials_in_grid": len(ev["grid"]),
        "trials_total": ledger,
        "oos_sharpe": bt.sharpe(oos),
        "psr": bt.psr(oos),
        "dsr": bt.dsr(oos, srs),
        "haircut_sharpe": bt.haircut_sharpe(bt.sharpe(oos), len(oos), ledger),
        "pbo": None if pbo is None else pbo["pbo"],
        "lookahead_suspicious": look["suspicious"],
        "complete": True,
    }


class RunParams(BaseModel):
    pass


@analysis(
    id="TR-001",
    version="1.1.0",
    title="Backtest (validated)",
    description=(
        "Vectorised run of the strategy's template and parameters: 1-bar signal shift, warm-up "
        "exclusion, fees + slippage on traded notional; IS/OOS split by time. The validation block "
        "(OOS Sharpe, PSR, DSR over the trial ledger, haircut Sharpe, PBO on the grid, look-ahead "
        "check) is computed first and returned with the results; the equity curve is withheld when "
        "the look-ahead check fails."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def backtest_run(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    validation = _validation(strat, ev, close)
    target = _target(strat["spec"], close, ev["params"])
    frame = bt.run_vectorised(close, target, ev["costs"], ev["warmup"])
    is_frame, oos_frame = frame.loc[frame.index < ev["cut"]], frame.loc[frame.index >= ev["cut"]]
    show = not validation["lookahead_suspicious"]
    nav = (1 + frame["net"]).cumprod()
    bench = (1 + frame["asset"]).cumprod()
    step = max(len(nav) // 400, 1)
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "hypothesis": strat["hypothesis"],
            "params": ev["params"],
            "split": ev["cut"].date().isoformat(),
            "validation": validation,
            "in_sample": bt.performance(is_frame),
            "out_of_sample": bt.performance(oos_frame),
            "trades": bt.trade_metrics(bt.trades_of(oos_frame)),
            "dates": [d.date().isoformat() for d in nav.index[::step]] if show else [],
            "lines": {
                "strategy": [float(v) for v in nav.to_numpy()[::step]],
                "buy_and_hold": [float(v) for v in bench.to_numpy()[::step]],
            }
            if show
            else {},
            "note": None if show else "Equity curve withheld: the look-ahead check failed.",
        },
    )


@analysis(
    id="TR-002",
    version="1.0.0",
    title="Parameter grid & sensitivity",
    description=(
        "Grid search: IS and OOS Sharpe for every configuration (heatmap for two parameters), the "
        "top-5 IS configurations with their OOS metrics (TEMA notebook), and the neighbourhood "
        "stability score = mean OOS Sharpe of the chosen cell's grid neighbours / its own."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def grid_sensitivity(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    grid, is_sr, oos_sr = ev["grid"], ev["is_sr"], ev["oos_sr"]
    order = is_sr.sort_values(ascending=False).index[:5]
    trials = ev["trials"]
    top5 = []
    for i in order:
        oos_frame = pd.DataFrame({"net": trials.loc[trials.index >= ev["cut"], i]})
        r = oos_frame["net"]
        nav = (1 + r).cumprod()
        top5.append(
            {
                **{k: grid[i][k] for k in grid[i]},
                "is_sharpe": float(is_sr[i]),
                "oos_sharpe": float(oos_sr[i]),
                "oos_return": float(nav.iloc[-1] - 1),
                "oos_max_drawdown": float((nav / nav.cummax() - 1).min()),
            }
        )
    keys = list(strat["spec"]["grid"])
    heat = None
    if len(keys) == 2:
        a, b = keys
        rows_v = sorted({g[a] for g in grid})
        cols_v = sorted({g[b] for g in grid})
        lookup = {(g[a], g[b]): i for i, g in enumerate(grid)}
        heat = {
            "rows": [f"{a}={v:g}" for v in rows_v],
            "columns": [f"{b}={v:g}" for v in cols_v],
            "values": [
                [_safe(oos_sr[lookup[(ra, cb)]]) if (ra, cb) in lookup else None for cb in cols_v]
                for ra in rows_v
            ],
            "unit": "number",
            "best": [],
        }
    c = grid[ev["chosen"]]
    neighbours = [
        i
        for i, g in enumerate(grid)
        if i != ev["chosen"] and _is_neighbour(g, c, strat["spec"]["grid"])
    ]
    own = float(oos_sr[ev["chosen"]])
    stability = (
        float(np.mean([oos_sr[i] for i in neighbours]) / own)
        if neighbours and own not in (0.0,) and math.isfinite(own)
        else None
    )
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "configurations": len(grid),
            "rows": top5,
            "chosen": c,
            "chosen_oos_sharpe": own,
            "neighbourhood_stability": stability,
            **({"grid": heat} if heat else {}),
        },
    )


def _safe(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _is_neighbour(g: dict[str, float], c: dict[str, float], ranges: dict[str, list[float]]) -> bool:
    steps = 0
    for k, values in ranges.items():
        vals = sorted(float(v) for v in values)
        d = abs(vals.index(g[k]) - vals.index(c[k]))
        if d > 1:
            return False
        steps += d
    return steps >= 1


@analysis(
    id="TR-003",
    version="1.0.0",
    title="Validation protocols",
    description=(
        "IS/OOS split with indicators warmed on strictly past data; rolling walk-forward selection "
        "(best IS Sharpe per block, applied to the next block) and the same walk-backward in "
        "reversed time — inconsistent forward/backward Sharpe ratios are evidence of overfitting "
        "(AFML-12)."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def protocols(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    fwd = bt.walk_forward(ev["trials"], 5)
    back = bt.walk_forward(ev["trials"], 5, backward=True)
    gap = abs(fwd["sharpe"] - back["sharpe"])
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "split": ev["cut"].date().isoformat(),
            "is_sharpe_chosen": float(ev["is_sr"][ev["chosen"]]),
            "oos_sharpe_chosen": float(ev["oos_sr"][ev["chosen"]]),
            "walk_forward_sharpe": fwd["sharpe"],
            "walk_backward_sharpe": back["sharpe"],
            "forward_backward_gap": gap,
            "consistent": bool(gap < 0.5),
            "walk_forward_choices": [ev["grid"][i] for i in fwd["chosen"]],
        },
    )


class CpcvParams(BaseModel):
    groups: int = Field(6, ge=3, le=12)
    test_groups: int = Field(2, ge=1, le=5)
    embargo: int = Field(5, ge=0, le=60)


@analysis(
    id="TR-014",
    version="1.0.0",
    title="CPCV backtest paths",
    description=(
        "Combinatorial purged CV over the grid: N groups, k test groups → C(N,k) splits and "
        "φ = k/N·C(N,k) paths; embargo after each tested group; in each split the best training "
        "configuration is evaluated on the test groups; Sharpe per path, its dispersion, average "
        "inter-path correlation and the variance of the mean path Sharpe."
    ),
    applies_to=STRATEGY,
    params=CpcvParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def cpcv(inputs: AnalysisInputs, params: CpcvParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    if params.test_groups >= params.groups:
        raise InsufficientDataError("k must be smaller than N")
    ev = _evaluate(strat["spec"], _close(prices))
    out = bt.cpcv_paths(ev["trials"], params.groups, params.test_groups, params.embargo)
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            **out,
            "rows": [{"path": i + 1, "sharpe": s} for i, s in enumerate(out["path_sharpes"])],
        },
    )


@analysis(
    id="TR-007",
    version="1.0.0",
    title="Overfitting control",
    description=(
        "OOS Sharpe of the chosen configuration with PSR (SR* = 0), Deflated Sharpe (SR₀ from the "
        "expected maximum over the trial ledger), Bonferroni haircut Sharpe; the ledger counts "
        "every distinct grid configuration ever run on this strategy and cannot be reset."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def overfitting(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    return AnalysisOutput("table", {"label": LABEL, **_validation(strat, ev, close)})


@analysis(
    id="TR-009",
    version="1.0.0",
    title="CSCV & PBO",
    description=(
        "Combinatorially symmetric CV over 16 row blocks of the grid's trial matrix: logit of the "
        "OOS rank of the IS-best configuration for all C(16,8) splits, PBO = P(logit ≤ 0), OOS-vs-"
        "IS Sharpe degradation slope, probability of an OOS loss."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def cscv(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    ev = _evaluate(strat["spec"], _close(prices))
    out = bt.pbo(ev["trials"], 16)
    hist, edges = np.histogram(out.pop("logits"), bins=20)
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            **out,
            "histogram": [
                {"from": float(a), "to": float(b), "count": int(c)}
                for a, b, c in zip(edges[:-1], edges[1:], hist, strict=True)
            ],
        },
    )


@analysis(
    id="TR-008",
    version="1.1.0",
    title="Seven sins checklist",
    description=(
        "Automatic checks before results: survivorship (single listed symbol: flagged as such), "
        "look-ahead (signal shift + delay test), storytelling (hypothesis note), data snooping "
        "(trial ledger → DSR), transaction costs (on), outliers (OOS Sharpe without the top-5 days "
        "and without the top-5 trades), shorting (borrow assumption when shorts are allowed)."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def seven_sins(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    val = _validation(strat, ev, close)
    target = _target(strat["spec"], close, ev["params"])
    frame = bt.run_vectorised(close, target, ev["costs"], ev["warmup"])
    oos = frame.loc[frame.index >= ev["cut"]]
    r = oos["net"].copy()
    without_days = r.drop(r.nlargest(5).index)
    trades = bt.trades_of(oos)
    pf_wo = None
    if len(trades) > 5:
        kept = trades.drop(trades["return"].nlargest(5).index)
        pf_wo = float(np.prod(1 + kept["return"]) - 1)
    rows = [
        {
            "sin": "survivorship",
            "status": "warn",
            "detail": "Single symbol listed today; delisted alternatives not considered",
        },
        {
            "sin": "look-ahead",
            "status": "fail" if val["lookahead_suspicious"] else "pass",
            "detail": "Signals shifted one bar; delay test",
        },
        {
            "sin": "storytelling",
            "status": "pass" if len(strat["hypothesis"]) >= 20 else "fail",
            "detail": strat["hypothesis"][:120],
        },
        {
            "sin": "data snooping",
            "status": "pass" if (val["dsr"] or 0) >= 0.95 else "warn",
            "detail": f"{val['trials_total']} trials; DSR {val['dsr']:.2f}"
            if val["dsr"] is not None and math.isfinite(val["dsr"])
            else f"{val['trials_total']} trials",
        },
        {
            "sin": "transaction costs",
            "status": "pass",
            "detail": f"{ev['costs'].fee_bps:g} bp fees + {ev['costs'].slippage_bps:g} bp slippage",
        },
        {
            "sin": "outliers",
            "status": "pass" if bt.sharpe(without_days.to_numpy()) > 0 else "warn",
            "detail": (
                f"OOS Sharpe without top-5 days {bt.sharpe(without_days.to_numpy()):.2f}; "
                f"return without top-5 trades {pf_wo if pf_wo is None else round(pf_wo, 4)}"
            ),
        },
        {
            "sin": "shorting",
            "status": "warn" if strat["spec"]["short"] else "pass",
            "detail": "Shorts assume borrow at 50 bp/yr (event-driven engine)"
            if strat["spec"]["short"]
            else "Long-only",
        },
    ]
    return AnalysisOutput(
        "table",
        {"label": LABEL, "rows": rows, "failures": sum(r["status"] == "fail" for r in rows)},
    )


@analysis(
    id="TR-016",
    version="1.1.0",
    title="Backtest statistics",
    description=(
        "AFML-14 statistics on the OOS period: general characteristics (exposure, long ratio, "
        "turnover, correlation to the asset), performance (return, CAGR, hit ratio), runs (HHI of "
        "positive returns, drawdowns, time under water), efficiency (Sharpe, Sortino, Calmar, "
        "Omega, tail ratio), implementation costs, plus the TEMA-notebook trade analytics (TR-005)."
    ),
    applies_to=STRATEGY,
    params=RunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def statistics(inputs: AnalysisInputs, params: RunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    target = _target(strat["spec"], close, ev["params"])
    frame = bt.run_vectorised(close, target, ev["costs"], ev["warmup"])
    oos = frame.loc[frame.index >= ev["cut"]]
    trades = bt.trades_of(oos)
    rows = (
        [
            {
                "entry": pd.Timestamp(t["entry"]).date().isoformat(),
                "exit": pd.Timestamp(t["exit"]).date().isoformat(),
                "side": t["side"],
                "return": t["return"],
                "bars": t["bars"],
                "mae": t["mae"],
                "mfe": t["mfe"],
            }
            for t in trades.to_dict("records")
        ]
        if not trades.empty
        else []
    )
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            **bt.performance(oos),
            "trade_metrics": bt.trade_metrics(trades),
            "rows": rows[-50:],
        },
    )


class RiskParams(BaseModel):
    target_sharpe: float = Field(1.0, ge=0, le=5)


@analysis(
    id="TR-017",
    version="1.1.0",
    title="Strategy risk",
    description=(
        "Binary-bet view of the OOS trades (AFML-15): precision p, average win/loss, bets per year;"
        " Sharpe implied by (p, π±, n); implied precision for the target Sharpe; probability of "
        "strategy failure P[p < p_θ] by bootstrap of the observed bets."
    ),
    applies_to=STRATEGY,
    params=RiskParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def strategy_risk(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    frame = bt.run_vectorised(
        close,
        _target(strat["spec"], close, ev["params"]),
        ev["costs"],
        ev["warmup"],
    )
    oos = frame.loc[frame.index >= ev["cut"]]
    trades = bt.trades_of(oos)
    if len(trades) < 5:
        raise InsufficientDataError(f"Strategy risk needs ≥ 5 OOS trades, got {len(trades)}")
    years = len(oos) / 252
    out = bt.failure_probability(
        trades["return"].to_numpy(), len(trades) / years, params.target_sharpe
    )
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "bets_per_year": len(trades) / years,
            "target_sharpe": params.target_sharpe,
            **out,
        },
    )


class SyntheticParams(BaseModel):
    forecast: float = Field(0.0, ge=-5, le=5, description="Expected level in σ units")
    half_life: float | None = Field(None, gt=0, le=500, description="Default: fitted on the price")


@analysis(
    id="TR-015",
    version="1.0.0",
    title="Synthetic OU trading rules",
    description=(
        "AFML-13: discrete OU fitted to the z-scored log price (TS-009 estimator), paths simulated "
        "from 0 towards the forecast, Sharpe of exits at profit-take × stop-loss (σ units) over a "
        "100-bar horizon; heat map and the best rule; σ-invariant (normalised units)."
    ),
    applies_to=STRATEGY,
    params=SyntheticParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def synthetic_rules(inputs: AnalysisInputs, params: SyntheticParams) -> AnalysisOutput:
    from fda_core.analyses.ts_models import half_life, ou_fit

    _, prices = _strategy(inputs)
    logp = np.log(_close(prices).to_numpy())
    z = (logp - logp.mean()) / logp.std()
    fit = ou_fit(z)
    fitted = half_life(fit["phi"])
    hl = params.half_life or fitted or 50.0
    source = (
        "parameter"
        if params.half_life
        else "fitted"
        if fitted
        else "default 50 bars (series not mean-reverting)"
    )
    pts = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    sls = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    grid = bt.ou_rule_map(params.forecast, float(hl), 1.0, pts, sls)
    i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "half_life": float(hl),
            "half_life_source": source,
            "phi": fit["phi"],
            "forecast": params.forecast,
            "best": {"take_profit": pts[i], "stop_loss": sls[j], "sharpe": float(grid[i, j])},
            "grid": {
                "rows": [f"PT {p}σ" for p in pts],
                "columns": [f"SL {s}σ" for s in sls],
                "values": [[float(v) for v in row] for row in grid],
                "unit": "number",
                "best": [],
            },
        },
    )


class EventParams(BaseModel):
    capital: float = Field(100_000, gt=0)
    participation: float = Field(0.1, gt=0, le=1)
    borrow_bps: float = Field(50.0, ge=0, le=2000)


@analysis(
    id="TR-004",
    version="1.0.0",
    title="Event-driven run",
    description=(
        "Orders from the target at each close, filled at the next open (adjusted) with slippage "
        "and fees, capped at participation × bar volume (partial fills carry over), borrow cost on "
        "shorts, cash accounting; compared with the vectorised close-to-close run."
    ),
    applies_to=STRATEGY,
    params=EventParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def event_driven(inputs: AnalysisInputs, params: EventParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    close = _close(prices)
    ev = _evaluate(strat["spec"], close)
    factor = prices["adj_close"] / prices["close"]
    bars = pd.DataFrame(
        {
            "open": prices["open"] * factor,
            "close": close,
            "volume": prices.get("volume", pd.Series(np.inf, index=prices.index)),
        }
    )
    target = _target(strat["spec"], close, ev["params"])
    evt = bt.run_event_driven(
        bars,
        target,
        ev["costs"],
        params.capital,
        params.participation,
        params.borrow_bps,
        ev["warmup"],
    )
    vec = bt.run_vectorised(close, target, ev["costs"], ev["warmup"])
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "event_driven": bt.performance(
                pd.DataFrame(
                    {
                        "net": evt["net"],
                        "position": np.sign(evt["shares"]),
                        "cost": 0.0,
                        "asset": vec["asset"].reindex(evt.index).fillna(0.0),
                    }
                )
            ),
            "vectorised": bt.performance(vec),
            "final_equity": float(evt["equity"].iloc[-1]),
            "max_shares": float(evt["shares"].abs().max()),
        },
    )
