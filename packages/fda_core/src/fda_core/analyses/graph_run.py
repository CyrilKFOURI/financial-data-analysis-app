"""TR-027 Strategy graph run (feature 321): the Results tab of the strategy builder.

Subject `strategy` with template `graph`: the graph runs on the strategy's stored daily bars
(adjusted OHLC) and returns the clone's Results sections — run configuration, performance
summary, equity curve with trade markers — plus the trade list. Validation (DSR, PBO,
robustness) comes from the lab's analyses on the same strategy.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from fda_core import blocks as bk
from fda_core import robustness as rb
from fda_core.analyses.backtesting import LABEL, STRATEGY, _strategy
from fda_core.analyses.price import PRICE_SUBJECTS, _prices
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis


class GraphRunParams(BaseModel):
    pass


def graph_payload(graph: dict[str, Any], prices: pd.DataFrame) -> dict[str, Any]:
    """The Results sections of a graph run on these bars."""
    result = bk.run(graph, prices)
    sim = result.simulation
    portfolio = next(b for b in graph["blocks"] if b["type"] == "PortfolioSimulation")
    cash = float((portfolio.get("params") or {}).get("InitialCash", 100_000))
    periods = rb.bars_per_year(prices.index)
    step = max(len(sim.equity) // 600, 1)
    markers = [
        {
            "date": t[date_key],
            "side": side,
            "price": t[price_key],
            "exit": t["exit"] if side == "sell" else None,
        }
        for t in sim.trades
        for side, date_key, price_key in (
            ("buy", "entry_date", "entry_price"),
            ("sell", "exit_date", "exit_price"),
        )
    ]
    return {
        "label": LABEL,
        "run_configuration": {
            "interval": "1d",
            "run_period_start": prices.index[0].date().isoformat(),
            "run_period_end": prices.index[-1].date().isoformat(),
            "bars": len(prices),
            "blocks": len(graph["blocks"]),
            "wires": len(graph["wires"]),
        },
        "performance_summary": bk.performance(sim, cash, periods),
        "dates": [d.date().isoformat() for d in sim.equity.index[::step]],
        "lines": {"equity": [float(v) for v in sim.equity.to_numpy()[::step]]},
        "markers": markers,
        "trades": sim.trades[-200:],
    }


@analysis(
    id="TR-027",
    version="1.0.0",
    title="Strategy graph run",
    description=(
        "Runs a strategy-builder graph on the strategy's daily bars: every block in topological "
        "order, then the Portfolio Simulation (long-only; entries and exits at the close of the "
        "signal bar; stop checked against the previous bar's level, take profit at the "
        "risk/reward multiple; fees per unit and slippage on every fill). Returns the run "
        "configuration, the performance summary (trades, win rate, realized P&L, fees, max "
        "drawdown and its duration, profit factor, max consecutive losses, Sharpe, skewness, "
        "excess kurtosis, final equity), the equity curve and the trade markers."
    ),
    applies_to=STRATEGY,
    params=GraphRunParams,
    result_kind="table",
    requires={"strategy", "prices"},
    budget_ms=60_000,
)
def graph_run(inputs: AnalysisInputs, params: GraphRunParams) -> AnalysisOutput:
    strat, prices = _strategy(inputs)
    spec = strat["spec"]
    if spec.get("template") != "graph" or not isinstance(spec.get("graph"), dict):
        raise InsufficientDataError("This strategy is not built from blocks")
    payload = graph_payload(spec["graph"], prices)
    payload["run_configuration"] |= {"strategy_id": strat.get("id"), "symbol": spec["symbol"]}
    return AnalysisOutput("table", payload)


class PreviewParams(BaseModel):
    graph: dict[str, Any] = Field(description="The graph on the canvas (blocks and wires)")


@analysis(
    id="TR-028",
    version="1.0.0",
    title="Strategy graph preview",
    description=(
        "Runs the graph currently on the canvas, saved or not, on the asset's daily bars: the "
        "same execution and Results as TR-027. The Candlestick and Portfolio Simulation trade "
        "the asset of the subject."
    ),
    applies_to=PRICE_SUBJECTS,
    params=PreviewParams,
    result_kind="table",
    requires={"prices"},
    budget_ms=60_000,
)
def graph_preview(inputs: AnalysisInputs, params: PreviewParams) -> AnalysisOutput:
    prices = _prices(inputs)
    if len(prices) < 60:
        raise InsufficientDataError(f"Needs ≥ 60 daily bars, got {len(prices)}")
    problems = bk.validate(params.graph)
    if problems:
        raise InsufficientDataError("The graph cannot run: " + "; ".join(problems))
    return AnalysisOutput("table", graph_payload(params.graph, prices))
