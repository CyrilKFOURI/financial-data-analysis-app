"""Paper trading (ANALYTICS_CATALOG TR-040) and execution analytics (TR-042); module M29.

Input `paper` = {account, orders, costs, bars}: the stored orders of a paper account and the
daily bars of every symbol they trade. Everything is replayed by `fda_core.paper.simulate`, so
the statement reconciles to the fills to the cent. Simulated: nothing reaches a broker.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.paper import execution_report, reconcile, simulate
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

PAPER: set[Any] = {"paper"}
CAVEATS = [
    "Daily bars: orders decided at a session's close fill from the next session (no intraday).",
    "Unadjusted OHLC; dividends and splits after an order are not applied to paper positions.",
    "Cash account: no margin, no shorting; costs = fee (bps, minimum) + half-spread + "
    "square-root impact on 20-day ADV.",
]


def _paper(inputs: AnalysisInputs) -> dict[str, Any]:
    data = inputs.data.get("paper")
    if not isinstance(data, dict):
        raise InsufficientDataError("Paper account data unavailable")
    return data


def _costs_text(c: dict[str, float]) -> str:
    return (
        f"fee {c['fee_bps']:g} bp (min {c['min_fee']:g}) · "
        f"half-spread {c['half_spread_bps']:g} bp · "
        f"impact η {c['impact_eta']:g} × σ × √(Q/ADV₂₀)"
    )


class StatementParams(BaseModel):
    rows: int = Field(30, ge=5, le=500, description="Daily statement rows shown (latest)")


@analysis(
    id="TR-040",
    version="1.0.1",
    title="Paper account",
    description=(
        "Paper account replayed from its orders over the stored daily bars: fills (next-session "
        "rules per order type, DAY/GTC), positions at average cost, cash, realised and unrealised "
        "P&L, fees, equity curve and daily statement; reconciliation of the book equity to the "
        "fills to the cent. Simulated — nothing reaches a broker."
    ),
    applies_to=PAPER,
    params=StatementParams,
    result_kind="series",
    requires={"paper"},
)
def paper_account(inputs: AnalysisInputs, params: StatementParams) -> AnalysisOutput:
    data = _paper(inputs)
    account = data["account"]
    start = float(account["starting_cash"])
    sim = simulate(start, data["orders"], data["bars"], data["costs"])
    rec = reconcile(start, sim)
    st = sim.statement
    equity = st["equity"] if not st.empty else pd.Series(dtype=float)
    r = equity.pct_change().dropna()
    dd = equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
    fills_by_order: dict[int, list[dict[str, Any]]] = {}
    for f in sim.fills:
        fills_by_order.setdefault(f.order_id, []).append(
            {"date": f.date.isoformat(), "quantity": f.quantity, "price": f.price, "fee": f.fee}
        )
    rows = {int(o["id"]): o for o in data["order_rows"]}
    orders = [
        {
            "id": s.order.id,
            "decided": s.order.decision_date.isoformat(),
            "symbol": s.order.symbol,
            "side": s.order.side,
            "quantity": s.order.quantity,
            "type": s.order.order_type,
            "limit": s.order.limit_price,
            "stop": s.order.stop_price,
            "trail": s.order.trail_pct,
            "tif": s.order.tif,
            "status": s.status,
            "filled": s.filled,
            "fill_price": (fills_by_order.get(s.order.id) or [{}])[0].get("price"),
            "fill_date": (fills_by_order.get(s.order.id) or [{}])[0].get("date"),
            "reason": s.reason or None,
            "source": s.order.source,
            "note": rows.get(s.order.id, {}).get("note"),
        }
        for s in sorted(sim.orders, key=lambda s: s.order.id, reverse=True)
    ]
    unrealised = sum(p["unrealised"] for p in sim.positions.values())
    end_equity = rec["equity_book"]
    statement = (
        [
            {"date": d.isoformat(), **{k: float(v) for k, v in row.items()}}
            for d, row in st.tail(params.rows).iloc[::-1].iterrows()
        ]
        if not st.empty
        else []
    )
    return AnalysisOutput(
        "series",
        {
            "account": account["name"],
            "currency": account["currency"],
            "starting_cash": start,
            "equity": end_equity,
            "cash": rec["cash_book"],
            "total_return": end_equity / start - 1,
            "realised_pnl": round(sim.realised, 2),
            "unrealised_pnl": round(unrealised, 2),
            "fees": round(sim.fees, 2),
            "volatility": float(r.std(ddof=1) * math.sqrt(252)) if len(r) > 2 else None,
            "max_drawdown": float(dd.min()) if not dd.empty else None,
            "reconciliation": rec,
            "dates": [d.isoformat() for d in st.index] if not st.empty else [],
            "lines": {
                "equity": st["equity"].tolist() if not st.empty else [],
                "cash": st["cash"].tolist() if not st.empty else [],
            },
            "positions": [{"symbol": s, **p} for s, p in sorted(sim.positions.items())],
            "orders": orders,
            "statement": statement,
            "cost_model": _costs_text(account["costs"]),
            "caveats": CAVEATS,
            "model_output": "Simulated paper account; not a broker statement.",
        },
    )


@analysis(
    id="TR-042",
    version="1.0.0",
    title="Execution analytics",
    description=(
        "Per fill and value-weighted: implementation shortfall vs the decision price = delay "
        "(next open vs decision close) + execution (fill vs open) + fees, in bps, positive = "
        "cost; slippage vs the cost model; fill vs the daily typical price (H+L+C)/3 as a VWAP "
        "proxy; fill rate of completed orders; shortfall by weekday (daily bars: no time of day)."
    ),
    applies_to=PAPER,
    params=StatementParams,
    result_kind="table",
    requires={"paper"},
)
def execution_analytics(inputs: AnalysisInputs, params: StatementParams) -> AnalysisOutput:
    data = _paper(inputs)
    sim = simulate(
        float(data["account"]["starting_cash"]), data["orders"], data["bars"], data["costs"]
    )
    if not sim.orders:
        raise InsufficientDataError("No orders yet: place a paper order first")
    report = execution_report(sim.fills, sim.orders)
    fills = report["fills"][-params.rows :][::-1]
    shortfall = np.array([f["shortfall_bps"] for f in report["fills"]])
    return AnalysisOutput(
        "table",
        {
            **{k: v for k, v in report.items() if k != "fills"},
            "fills": fills,
            "shortfall_dispersion_bps": float(shortfall.std(ddof=1))
            if len(shortfall) > 1
            else None,
            "source": "paper fills (simulated: slippage equals the model by construction)",
            "pending": "Broker executions (IBKR read-only sync) await the ib_async decision.",
        },
    )
