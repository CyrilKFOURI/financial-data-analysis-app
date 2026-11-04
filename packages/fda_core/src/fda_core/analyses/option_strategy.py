"""Option strategies (ANALYTICS_CATALOG TR-030, TR-031; module M28).

TR-030 builds a multi-leg position from a preset or typed legs. On the stored chain, legs snap to
listed strikes of the nearest expiry and are priced at mid (or natural: long at the ask, short at
the bid) with their own solved IV; without a chain (or `source = model`) they are priced by BSM at
one volatility. TR-031 runs systematic option rules on synthetic chains (BSM at an IV proxy, flat
smile), always as a 12-configuration set so DSR and PBO come with the result. V5: outputs are
labelled model outputs, not recommendations.
"""

import math
import re
from dataclasses import replace
from typing import Any, Literal, cast, get_args

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core import backtest as bt
from fda_core import option_strategies as os_
from fda_core.analyses.derivatives import _chain, _implied_history, _rates
from fda_core.analyses.price import _field, _prices
from fda_core.option_strategies import Leg
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

ASSETS: set[Any] = {"asset"}
MULTIPLIER = 100
LEG_RE = re.compile(r"^([+-]?\d*\.?\d*)\s*([CPS])\s*(\d+\.?\d*)?\s*(?:@\s*(\d+))?$", re.I)


def parse_legs(text: str, spot: float, days: float, vol: float) -> list[Leg]:
    """`+1 C 105 @30; -2 P 95; +1 S` → legs (quantity sign = side; S = underlying at the spot;
    strike omitted = at the money; @days defaults to the strategy horizon)."""
    legs: list[Leg] = []
    for raw in re.split(r"[;,\n]", text):
        token = raw.strip()
        if not token:
            continue
        m = LEG_RE.match(token)
        if not m:
            raise InsufficientDataError(f"Cannot read leg {token!r} (expected e.g. '+1 C 105 @30')")
        qty_text, kind_code, strike_text, days_text = m.groups()
        qty = (
            float(qty_text)
            if qty_text not in ("", "+", "-", None)
            else (-1.0 if qty_text == "-" else 1.0)
        )
        if qty == 0:
            continue
        kind = {"C": "call", "P": "put", "S": "stock"}[kind_code.upper()]
        strike = spot if kind == "stock" or not strike_text else float(strike_text)
        legs.append(
            Leg(
                cast(os_.LegKind, kind),
                1 if qty > 0 else -1,
                strike,
                float(days_text) if days_text else days,
                abs(qty),
                0.0,
                vol,
            )
        )
    if not legs:
        raise InsufficientDataError("No legs given")
    if len(legs) > 8:
        raise InsufficientDataError("At most 8 legs")
    return legs


def _hist_vol(inputs: AnalysisInputs) -> tuple[float, float, pd.Series]:
    close = _prices(inputs)[_field(inputs)].astype(float).dropna()
    r = np.log(close).diff().dropna()
    if len(r) < 60:
        raise InsufficientDataError("Need ≥ 60 daily returns for the historical volatility")
    year = r.iloc[-252:]
    return float(year.std(ddof=1) * math.sqrt(252)), float(r.iloc[-1260:].mean() * 252), close


def snap_to_chain(
    legs: list[Leg], frame: pd.DataFrame, pricing: str
) -> tuple[list[Leg], list[dict[str, Any]]]:
    """Each option leg → nearest listed strike of the expiry nearest its horizon (quotes with a
    solved IV and a positive mid only); entry at mid or natural; leg vol = that quote's IV."""
    usable = frame.loc[frame["iv"].notna() & (frame["mid"] > 0)]
    if usable.empty:
        raise InsufficientDataError("No usable quotes (solved IV and positive mid) in the chain")
    expiries = usable.groupby("expiry")["days"].first()
    out: list[Leg] = []
    rows: list[dict[str, Any]] = []
    for leg in legs:
        if leg.kind == "stock":
            out.append(leg)
            rows.append(
                {"kind": "stock", "side": leg.side, "quantity": leg.quantity, "strike": leg.strike}
            )
            continue
        expiry = (expiries - leg.days).abs().idxmin()
        quotes = usable.loc[(usable["expiry"] == expiry) & (usable["kind"] == leg.kind)]
        if quotes.empty:
            raise InsufficientDataError(f"No {leg.kind} quotes for {expiry}")
        row = quotes.iloc[int((quotes["strike"] - leg.strike).abs().to_numpy().argmin())]
        bid, ask, mid = float(row["bid"] or 0), float(row["ask"] or 0), float(row["mid"])
        price = mid
        if pricing == "natural" and bid > 0 and ask > 0:
            price = ask if leg.side > 0 else bid
        snapped = replace(
            leg,
            strike=float(row["strike"]),
            days=float(row["days"]),
            premium=price,
            vol=float(row["iv"]),
        )
        out.append(snapped)
        rows.append(
            {
                "kind": leg.kind,
                "side": leg.side,
                "quantity": leg.quantity,
                "strike": snapped.strike,
                "expiry": str(pd.Timestamp(expiry).date()),
                "days": snapped.days,
                "bid": bid or None,
                "ask": ask or None,
                "premium": price,
                "iv": snapped.vol,
            }
        )
    return out, rows


def _priced_rows(legs: list[Leg]) -> list[dict[str, Any]]:
    return [
        {
            "kind": leg.kind,
            "side": leg.side,
            "quantity": leg.quantity,
            "strike": leg.strike,
            "days": leg.days if leg.kind != "stock" else None,
            "premium": leg.premium if leg.kind != "stock" else None,
            "iv": leg.vol if leg.kind != "stock" else None,
        }
        for leg in legs
    ]


def _grid(rows: list[str], columns: list[str], values: list[list[float]]) -> dict[str, Any]:
    return {"rows": rows, "columns": columns, "values": values, "unit": "number", "best": []}


class BuilderParams(BaseModel):
    preset: str = Field(
        "covered_call", description="One of the presets, ignored when legs are typed"
    )
    legs: str | None = Field(
        None, max_length=400, description="Typed legs, e.g. '+1 C 105 @30; -1 P 95'"
    )
    source: Literal["chain", "model"] = "chain"
    pricing: Literal["mid", "natural"] = "mid"
    days: int = Field(
        30, ge=1, le=730, description="Horizon in calendar days (nearest listed expiry)"
    )
    width: float = Field(
        0.05, gt=0, le=0.5, description="Inner strikes distance (fraction of spot)"
    )
    wing: float = Field(0.05, gt=0, le=0.5, description="Wing distance beyond the inner strikes")
    volatility: float | None = Field(
        None, gt=0, le=5, description="Model source: vol (default 1y HV)"
    )


@analysis(
    id="TR-030",
    version="1.0.0",
    title="Option strategy builder",
    description=(
        "Presets (covered call, protective put, collars incl. zero-cost, cash-secured & covered "
        "put, verticals, straddles/strangles, strip/strap, iron condor/butterfly, butterflies, "
        "calendar, diagonal, risk reversal, ratio) or typed legs, snapped to the stored chain (mid "
        "or natural, own IV) or priced by BSM. Payoff at the first expiry, P&L today and mid-life, "
        "breakevens, max profit/loss, probability of profit (risk-neutral at the ATM IV and "
        "physical at 1y HV with the 5y drift, both lognormal), Σ-leg Greeks, spot × time and spot ×"
        " vol P&L grids, scenarios and a Reg-T-style margin approximation. Per unit; ×100 per "
        "contract."
    ),
    applies_to=ASSETS,
    params=BuilderParams,
    result_kind="table",
    requires={"prices", "options"},
)
def strategy_builder(inputs: AnalysisInputs, params: BuilderParams) -> AnalysisOutput:
    r, q = _rates(inputs)
    hv, drift, close = _hist_vol(inputs)
    spot, as_of, frame = float(close.iloc[-1]), str(close.index[-1].date()), None
    source = params.source
    if source == "chain":
        try:
            chain_as_of, spot, frame = _chain(inputs)
            as_of = chain_as_of.isoformat()
        except InsufficientDataError:
            source = "model"
    atm_iv = hv
    if frame is not None:
        near = frame.loc[frame["iv"].notna() & (frame["mid"] > 0)].copy()
        if not near.empty:
            near["dist"] = (near["days"] - params.days).abs() * 1e6 + (near["strike"] - spot).abs()
            atm_iv = float(near.sort_values("dist")["iv"].iloc[:2].mean())
    vol = params.volatility or atm_iv
    preset = params.preset
    if params.legs:
        legs = parse_legs(params.legs, spot, params.days, vol)
        legs = [os_.priced(leg, spot, r, q) for leg in legs]
        preset = "custom"
    else:
        if preset not in get_args(os_.Preset):
            raise InsufficientDataError(
                f"Unknown preset {preset!r}; one of {', '.join(get_args(os_.Preset))}"
            )
        legs = os_.build_preset(preset, spot, vol, params.days, params.width, params.wing, r, q)
    if frame is not None and source == "chain":
        if (
            preset == "zero_cost_collar"
        ):  # re-solve on listed strikes: call premium closest to the put's
            legs, rows = snap_to_chain(legs[:2], frame, params.pricing)
            calls = frame.loc[
                (frame["kind"] == "call")
                & frame["iv"].notna()
                & (frame["mid"] > 0)
                & (frame["days"] == legs[1].days)
                & (frame["strike"] > spot)
            ]
            if calls.empty:
                raise InsufficientDataError("No listed calls above the spot for the collar")
            best = calls.iloc[int((calls["mid"] - legs[1].premium).abs().to_numpy().argmin())]
            more, more_rows = snap_to_chain(
                [Leg("call", -1, float(best["strike"]), legs[1].days, 1.0, 0.0, vol)],
                frame,
                params.pricing,
            )
            legs, rows = legs + more, rows + more_rows
        else:
            legs, rows = snap_to_chain(legs, frame, params.pricing)
    else:
        rows = _priced_rows(legs)
    for row, leg in zip(rows, legs, strict=True):
        if leg.kind != "stock":
            g = os_.aggregate_greeks([replace(leg, side=1, quantity=1.0)], spot, r, q)
            row["delta"] = g["delta"]

    t_first = os_.first_expiry(legs)
    s_grid = np.linspace(spot * 0.7, spot * 1.3, 121)
    with np.errstate(divide="ignore", invalid="ignore"):
        at_expiry = os_.expiry_pnl(legs, s_grid, r, q)
        today = os_.position_pnl(legs, s_grid, 0.0, r, q)
        half = os_.position_pnl(legs, s_grid, t_first / 2, r, q)
    ext = os_.extremes(legs, spot, r, q)
    shifts = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    times = sorted(
        {0.0, round(t_first * 0.25), round(t_first * 0.5), round(t_first * 0.75), t_first}
    )
    vols = [-0.10, -0.05, 0.0, 0.05, 0.10]

    def pnl_at(ds: float, elapsed: float, dv: float = 0.0) -> float:
        return float(os_.position_pnl(legs, np.array([spot * (1 + ds)]), elapsed, r, q, dv)[0])

    scenarios = [
        {
            "scenario": name,
            "spot_shift": ds,
            "vol_shift": dv,
            "days_passed": el,
            "pnl": pnl_at(ds, el, dv),
            "pnl_per_contract": pnl_at(ds, el, dv) * MULTIPLIER,
        }
        for name, ds, dv, el in [
            ("Spot −10 %", -0.10, 0.0, 0.0),
            ("Spot +10 %", 0.10, 0.0, 0.0),
            ("Vol +5 pts", 0.0, 0.05, 0.0),
            ("Vol −5 pts", 0.0, -0.05, 0.0),
            ("One week passes", 0.0, 0.0, min(7.0, t_first)),
            ("Crash: −20 %, vol +15 pts", -0.20, 0.15, 1.0),
        ]
    ]
    greeks = os_.aggregate_greeks(legs, spot, r, q)
    premium = os_.net_premium(legs)
    return AnalysisOutput(
        "table",
        {
            "as_of": as_of,
            "spot": spot,
            "r": r,
            "q": q,
            "preset": preset,
            "source": "stored chain" if source == "chain" and frame is not None else "BSM model",
            "pricing": params.pricing if source == "chain" and frame is not None else "model",
            "volatility_used": vol,
            "legs": rows,
            "net_premium": premium,
            "premium_kind": "credit" if premium > 0 else "debit" if premium < 0 else "zero cost",
            "first_expiry_days": t_first,
            "max_profit": ext["max_profit"],
            "max_loss": ext["max_loss"],
            "breakevens": os_.breakevens(s_grid, at_expiry),
            "pop_risk_neutral": os_.probability_of_profit(legs, spot, atm_iv, r - q, r, q),
            "pop_physical": os_.probability_of_profit(legs, spot, hv, drift, r, q),
            "pop_note": (
                f"Lognormal at the first expiry: risk-neutral at the ATM IV {atm_iv:.1%} with "
                f"drift r − q; physical at the 1y HV {hv:.1%} with the 5y mean drift {drift:.1%}."
            ),
            "greeks": greeks,
            "greeks_per_contract": {k: v * MULTIPLIER for k, v in greeks.items()},
            "vega_per_vol_point": greeks["vega"] / 100,
            "theta_per_day": greeks["theta"] / 365,
            "margin": {
                **os_.margin_estimate(legs, spot, r, q),
                "label": "approximation, not a broker figure",
            },
            "curve": {
                "spot": s_grid.tolist(),
                "expiry": at_expiry.tolist(),
                "today": today.tolist(),
                "half_life": half.tolist(),
            },
            "surface_time": _grid(
                [f"T+{int(t)}d" for t in times],
                [f"{s:+.0%}" for s in shifts],
                [[pnl_at(ds, t) for ds in shifts] for t in times],
            ),
            "surface_vol": _grid(
                [f"σ {dv:+.0%}" for dv in vols],
                [f"{s:+.0%}" for s in shifts],
                [[pnl_at(ds, 0.0, dv) for ds in shifts] for dv in vols],
            ),
            "scenarios": scenarios,
            "multiplier": MULTIPLIER,
            "presets": list(get_args(os_.Preset)),
            "model_output": "Labelled model output (BSM marks); not a recommendation.",
        },
    )


# ── TR-031 option strategy backtests ─────────────────────────────────────────

RULE_DOCS = {
    "covered_call_30d": "Long the underlying, short a 30Δ call, rolled at expiry",
    "put_write_atm": "Short an at-the-money put, rolled at expiry (CBOE PUT-style)",
    "short_strangle_16d": "Short 16Δ call and 16Δ put",
}


class RuleParams(BaseModel):
    rule: Literal["covered_call_30d", "put_write_atm", "short_strangle_16d"] = "put_write_atm"
    days: int = Field(30, ge=14, le=90, description="Tenor of each option cycle (calendar days)")
    managed: bool = Field(
        False, description="Exit at 50 % of the credit or 21 DTE (scaled to the tenor)"
    )
    cost_per_option: float = Field(0.02, ge=0, le=5, description="Per option and side, price units")
    years: int = Field(10, ge=2, le=30)
    iv_shift: float = Field(
        0.0, ge=-0.2, le=0.2, description="Vol points added to the IV proxy (−0.03 = −3 pts)"
    )


def _iv_proxy(inputs: AnalysisInputs, close: pd.Series) -> tuple[pd.Series, str]:
    history, source = _implied_history(inputs)
    if not history.empty and len(history.loc[history.index >= close.index[0]]) > 0.8 * len(close):
        return history, source
    realised = np.log(close).diff().rolling(21).std() * math.sqrt(252)
    return (
        realised.bfill(),
        "realised 21-day volatility (no implied history: the synthetic chain has no variance "
        "premium)",
    )


@analysis(
    id="TR-031",
    version="1.0.0",
    title="Option strategy backtests",
    description=(
        "Systematic option rules (30Δ covered call, ATM put-write, 16Δ short strangle; the chosen "
        "tenor and 45 (or 30) days; optional management at 50 % of the credit / 21 DTE scaled to "
        "the tenor) on synthetic chains: BSM at the IV proxy of each day (stored ATM IV, else the "
        "VIX-family index, else realised vol — labelled), flat smile, costs per option and side. "
        "The 12 rule × tenor × management configurations always run together: DSR over the 12 "
        "Sharpes and CSCV PBO come with the selected one."
    ),
    applies_to=ASSETS,
    params=RuleParams,
    result_kind="series",
    requires={"prices", "option_history", "vol_proxy"},
    budget_ms=30_000,
)
def strategy_backtest(inputs: AnalysisInputs, params: RuleParams) -> AnalysisOutput:
    r, q = _rates(inputs)
    close = _prices(inputs)[_field(inputs)].astype(float).dropna()
    close = close.loc[close.index >= close.index[-1] - pd.DateOffset(years=params.years)]
    if len(close) < 252:
        raise InsufficientDataError("Need ≥ 1 year of prices for an option rule backtest")
    proxy, iv_source = _iv_proxy(inputs, close)
    iv = (proxy + params.iv_shift).clip(lower=0.01)
    trials: dict[str, pd.DataFrame] = {}
    tenors = (params.days, 45 if params.days != 45 else 30)
    for rule in RULE_DOCS:
        for tenor in tenors:
            for managed in (False, True):
                key = f"{rule}|{tenor}|{'managed' if managed else 'held'}"
                trials[key] = os_.option_rule_backtest(
                    close,
                    iv,
                    rule,
                    r,
                    q,
                    tenor,
                    0.5 if managed else None,
                    round(tenor * 21 / 45) if managed else None,  # 21 DTE on a 45-day cycle
                    params.cost_per_option,
                )
    chosen = f"{params.rule}|{params.days}|{'managed' if params.managed else 'held'}"
    frame = trials[chosen]
    asset = close.pct_change().fillna(0.0)
    perf_frame = pd.DataFrame(
        {
            "net": frame["net"],
            "position": 1.0,
            "asset": asset,
            "cost": 0.0,
        }
    )
    perf = bt.performance(perf_frame, 252, r)
    returns = pd.DataFrame({k: v["net"] for k, v in trials.items()}).iloc[1:]
    srs = [
        float(returns[c].mean() / returns[c].std(ddof=1))
        for c in returns
        if returns[c].std(ddof=1) > 0
    ]
    net = frame["net"].to_numpy()[1:]
    overfit = bt.pbo(returns, 16)
    events = frame["event"]
    cycles = int(events.str.contains("open").sum())
    managed_exits = int(events.str.contains("managed").sum())
    nav = (1 + frame["net"]).cumprod()
    lower = os_.option_rule_backtest(
        close,
        (iv - 0.03).clip(lower=0.01),
        params.rule,
        r,
        q,
        params.days,
        0.5 if params.managed else None,
        round(params.days * 21 / 45) if params.managed else None,
        params.cost_per_option,
    )
    years = len(frame) / 252
    bh = (1 + asset).cumprod()
    table = [
        {
            "configuration": k.replace("|", " · "),
            "sharpe": bt.sharpe(v["net"].to_numpy()[1:]),
            "total_return": float((1 + v["net"]).prod() - 1),
            "max_drawdown": float(
                ((1 + v["net"]).cumprod() / (1 + v["net"]).cumprod().cummax() - 1).min()
            ),
        }
        for k, v in trials.items()
    ]
    return AnalysisOutput(
        "series",
        {
            "rule": params.rule,
            "rule_description": RULE_DOCS[params.rule],
            "configuration": chosen.replace("|", " · "),
            "iv_source": iv_source,
            "iv_shift": params.iv_shift,
            "pricing_bias": (
                "Flat smile at the proxy level: a VIX-type index is a variance-swap level above "
                "the ATM IV, and OTM calls trade below it (skew), so synthetic call premiums are "
                "overstated and OTM put premiums understated. Covered calls and strangles look "
                "better than with real chains; compare the −3 vol-point sensitivity."
            ),
            "sensitivity_minus_3pts": {
                "cagr": float((1 + lower["net"]).prod() ** (1 / years) - 1),
                "sharpe": bt.sharpe(lower["net"].to_numpy()[1:]),
            },
            "synthetic_chain": (
                "BSM at the IV proxy, flat smile, no early exercise: labelled approximation"
            ),
            "dates": [d.date().isoformat() for d in frame.index],
            "lines": {"strategy": nav.tolist(), "buy_and_hold": bh.tolist()},
            "performance": {
                k: v
                for k, v in perf.items()
                if k not in ("turnover_per_year", "costs_total", "exposure", "long_ratio")
            },
            "cycles": cycles,
            "managed_exits": managed_exits,
            "psr": bt.psr(net),
            "dsr": bt.dsr(net, srs),
            "trials": len(trials),
            "pbo": overfit["pbo"],
            "degradation_slope": overfit["degradation_slope"],
            "configurations": table,
            "model_output": (
                "Labelled model output on synthetic chains; DSR/PBO over the 12 configurations."
            ),
        },
    )
