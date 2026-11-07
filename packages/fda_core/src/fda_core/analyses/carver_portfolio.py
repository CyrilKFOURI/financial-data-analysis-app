"""TR-026 Multi-asset Carver book (Quant Rick bootcamp 2, the owner's review; feature 311).

Subject `portfolio`: its holdings are the basket and its weights the instrument weights. The
review's conclusion was "diversify across assets, not across indicators": on one equity index
the engine is crash insurance, on a basket of uncorrelated assets it carries the Sharpe. No
parameter is fitted (published Carver scalars, point-in-time FDM and IDM), so there is no
selection to deflate; the evidence is the comparison with the same basket bought and held and
the same basket sized by volatility without trend, by epoch and in crises.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core import carver
from fda_core import robustness as rb
from fda_core.analyses.carver_engine import CRISES
from fda_core.analyses.portfolio import PORTFOLIO, _assets, _f, _iso, _pf
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

LABEL = "Model output — backtest evidence, not a recommendation"
MIN_ASSETS = 2


class BookParams(BaseModel):
    vol_target: float = Field(
        0.20, gt=0, le=0.6, description="Annual volatility target of the book"
    )
    buffer: float = Field(0.10, ge=0, le=0.5, description="No-trade band, share of the position")
    max_leverage: float = Field(2.0, ge=0.5, le=4.0, description="Cap on gross exposure")


def _summary(name: str, frame: pd.DataFrame, periods: float) -> dict[str, Any]:
    s = rb.summary(frame["net"], periods)
    years = len(frame) / periods
    row: dict[str, Any] = {
        "series": name,
        "years": s["years"],
        "sharpe": s["sharpe"],
        "sharpe_se": s["sharpe_se"],
        "cagr": s["cagr"],
        "volatility": _f(frame["net"].std(ddof=1) * math.sqrt(periods)),
        "max_drawdown": s["max_drawdown"],
        "gross_exposure": _f(frame["position"].mean()),
        "turnover_per_year": _f(frame["cost"].sum() / max(years, 1e-9)),
    }
    for label, (a, b) in CRISES.items():
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        covered = frame.index[0] <= start and frame.index[-1] >= end
        window = frame.loc[(frame.index >= start) & (frame.index <= end), "net"]
        row[f"crisis_{label}"] = _f(float((1 + window).prod() - 1)) if covered else None
    return row


@analysis(
    id="TR-026",
    version="1.0.0",
    title="Multi-asset Carver book",
    description=(
        "The portfolio's holdings as a Carver basket: each instrument sized by its own forecasts "
        "(EWMAC, breakout, acceleration, skew; point-in-time FDM) at the book's volatility target, "
        "weighted by the portfolio weights and multiplied by a point-in-time instrument "
        "diversification multiplier IDM = 1/√(w'Hw) on sub-system return correlations (floored at "
        "0, cap 2.5), gross exposure capped. Compared with the basket bought and held and the "
        "basket sized by volatility without trend (same IDM method): Sharpe ± SE, CAGR, MaxDD, "
        "two epochs, crisis windows."
    ),
    applies_to=PORTFOLIO,
    params=BookParams,
    result_kind="table",
    requires={"portfolio"},
    budget_ms=90_000,
)
def carver_book(inputs: AnalysisInputs, params: BookParams) -> AnalysisOutput:
    pf = _pf(inputs)
    assets = _assets(pf)
    if len(assets) < MIN_ASSETS:
        raise InsufficientDataError(
            f"A Carver book needs at least {MIN_ASSETS} holdings besides cash, got {len(assets)}"
        )
    returns = pf.path.returns[assets].iloc[1:]
    warmup = carver.WARMUP + 1
    if len(returns) < warmup + 2 * 252:
        raise InsufficientDataError(
            f"Needs ≥ {warmup + 2 * 252} sessions (warm-up + two years), got {len(returns)}"
        )
    held = {h["symbol"]: float(h["weight"]) for h in pf.definition["holdings"]}
    raw_w = np.array([max(held.get(a, 0.0), 0.0) for a in assets], dtype=float)
    if raw_w.sum() <= 0:
        raise InsufficientDataError("The holdings have no positive weight")
    w = raw_w / raw_w.sum()
    periods = rb.bars_per_year(returns.index)
    cost = float(pf.definition.get("cost_bps") or 5.0) / 10_000

    positions, idm = carver.book_positions(
        returns,
        w,
        lambda price: carver.position(price, params.vol_target, params.buffer),
        params.max_leverage,
    )
    hold_positions, hold_idm = carver.book_positions(
        returns,
        w,
        lambda price: carver.vol_targeted_hold(price, params.vol_target),
        params.max_leverage,
    )
    static = pd.DataFrame(np.tile(w, (len(returns), 1)), index=returns.index, columns=assets)
    frames = {
        "carver_book": carver.run_book(returns, positions, cost, warmup),
        "vol_targeted_hold": carver.run_book(returns, hold_positions, cost, warmup),
        "buy_and_hold": carver.run_book(returns, static, cost, warmup),
    }
    sample = frames["carver_book"].index
    split = pd.Timestamp(sample[len(sample) // 2])

    instruments = []
    held_pos = positions.shift(1).iloc[warmup:]
    for i, a in enumerate(assets):
        sub = (held_pos[a] * returns[a].iloc[warmup:]).fillna(0.0)
        sr = rb.summary(sub, periods)["sharpe"]
        instruments.append(
            {
                "symbol": a,
                "weight": float(w[i]),
                "subsystem_sharpe": sr,
                "mean_position": _f(held_pos[a].mean()),
                "share_of_days_long": _f(float((held_pos[a] > 0).mean())),
            }
        )
    nav = {k: (1 + f["net"]).cumprod() for k, f in frames.items()}
    step = max(len(sample) // 400, 1)
    rows = [_summary(k, f, periods) for k, f in frames.items()]
    book, hold = rows[0], rows[1]
    return AnalysisOutput(
        "table",
        {
            "label": LABEL,
            "instruments": len(assets),
            "idm_last": _f(idm.iloc[-1]),
            "idm_mean": _f(idm.iloc[warmup:].mean()),
            "no_trend_idm_last": _f(hold_idm.iloc[-1]),
            "epoch_split": split.date().isoformat(),
            "rows": rows,
            "epochs": rb.epochs(frames, split, periods),
            "weights": instruments,
            "dates": _iso(sample[::step]),
            "lines": {
                "carver_book": [float(v) for v in nav["carver_book"].to_numpy()[::step]],
                "buy_and_hold": [float(v) for v in nav["buy_and_hold"].to_numpy()[::step]],
            },
            "note": (
                f"Book Sharpe {_fmt(book['sharpe'])} ± {_fmt(book['sharpe_se'])} against "
                f"{_fmt(hold['sharpe'])} for the same basket sized without trend: a gap smaller "
                "than the standard error is not evidence."
            ),
        },
    )


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.2f}"
