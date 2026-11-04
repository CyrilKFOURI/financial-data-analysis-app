"""Option strategy engine (M28, TR-030/TR-031): multi-leg positions, payoffs, P&L, Greeks,
probabilities, margin approximation, presets, and systematic option-rule backtests.

Conventions: a leg is (kind, side, strike, days, quantity, premium, vol); side +1 long / −1 short;
premiums are per unit at entry; a stock leg has strike = entry price. "Expiry" P&L is taken at the
first leg expiry: expired legs at intrinsic, longer-dated legs (calendar, diagonal) at BSM value
with their own volatility. Values are per unit of underlying (multiply by the contract size).
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats

from fda_core.options import Kind, bsm_greeks, bsm_price
from fda_core.registry import InsufficientDataError

Array = NDArray[np.float64]
LegKind = Literal["call", "put", "stock"]


@dataclass(frozen=True, slots=True)
class Leg:
    kind: LegKind
    side: int  # +1 long, −1 short
    strike: float  # entry price for a stock leg
    days: float = 30.0
    quantity: float = 1.0
    premium: float = 0.0
    vol: float = 0.2

    @property
    def signed(self) -> float:
        return self.side * self.quantity

    def intrinsic(self, s: Array) -> Array:
        if self.kind == "stock":
            return s
        if self.kind == "call":
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)

    def entry(self) -> float:
        return self.strike if self.kind == "stock" else self.premium

    def value(
        self,
        s: Array,
        days_left: float | Array,
        r: float,
        q: float,
        vol: float | Array | None = None,
    ) -> Array:
        """Mark (per unit, unsigned): intrinsic at/after expiry, BSM before (arrays broadcast)."""
        if self.kind == "stock":
            return self.intrinsic(s)
        left = np.asarray(days_left, dtype=float)
        v = self.vol if vol is None else vol
        with np.errstate(divide="ignore", invalid="ignore"):
            price = bsm_price(self.kind, s, self.strike, np.maximum(left, 1e-9) / 365, r, q, v)
        return np.asarray(np.where(left <= 0, self.intrinsic(s), price), dtype=float)

    def pnl(
        self,
        s: Array,
        elapsed: float | Array,
        r: float = 0.0,
        q: float = 0.0,
        vol_shift: float = 0.0,
        vol: float | Array | None = None,
    ) -> Array:
        """Signed P&L since entry after `elapsed` days, at `vol` (default: own vol) + shift."""
        v = np.maximum(
            (self.vol if vol is None else np.asarray(vol, dtype=float)) + vol_shift, 0.01
        )
        return self.signed * (
            self.value(s, self.days - np.asarray(elapsed, dtype=float), r, q, v) - self.entry()
        )


def first_expiry(legs: list[Leg]) -> float:
    days = [leg.days for leg in legs if leg.kind != "stock"]
    return min(days) if days else 0.0


def position_pnl(
    legs: list[Leg],
    s: Array,
    elapsed: float,
    r: float = 0.0,
    q: float = 0.0,
    vol_shift: float = 0.0,
) -> Array:
    total = np.zeros_like(s, dtype=float)
    for leg in legs:
        total = total + leg.pnl(s, elapsed, r, q, vol_shift)
    return total


def expiry_pnl(legs: list[Leg], s: Array, r: float = 0.0, q: float = 0.0) -> Array:
    return position_pnl(legs, s, first_expiry(legs), r, q)


def net_premium(legs: list[Leg]) -> float:
    """Cash at entry for the option legs: > 0 = credit received."""
    return float(sum(-leg.signed * leg.premium for leg in legs if leg.kind != "stock"))


def aggregate_greeks(legs: list[Leg], s: float, r: float, q: float) -> dict[str, float]:
    """Σ side × quantity × leg Greeks (stock: Δ = 1); vega per 1.00 vol, theta per year."""
    out = {k: 0.0 for k in ("delta", "gamma", "vega", "theta", "rho")}
    for leg in legs:
        if leg.kind == "stock":
            out["delta"] += leg.signed
            continue
        g = bsm_greeks(leg.kind, s, leg.strike, leg.days / 365, r, q, leg.vol)
        for k in out:
            out[k] += leg.signed * float(g[k])
    return out


def breakevens(s: Array, pnl: Array) -> list[float]:
    """Spots where the P&L curve crosses zero (linear interpolation on the grid)."""
    out: list[float] = []
    for i in range(len(s) - 1):
        a, b = float(pnl[i]), float(pnl[i + 1])
        if a == 0:
            out.append(float(s[i]))
        elif a * b < 0:
            out.append(float(s[i] - a * (s[i + 1] - s[i]) / (b - a)))
    return out


def upside_slope(legs: list[Leg]) -> float:
    """dP&L/dS as S → ∞ at the first expiry (calls and stock count, puts vanish)."""
    return float(sum(leg.signed for leg in legs if leg.kind in ("call", "stock")))


def extremes(
    legs: list[Leg], spot: float, r: float = 0.0, q: float = 0.0
) -> dict[str, float | None]:
    """Max profit / max loss at the first expiry over S ∈ [0, 10·S0]; None = unbounded as S → ∞."""
    grid = np.concatenate([np.linspace(0.0, spot * 3, 3001), [spot * 10]])
    with np.errstate(divide="ignore"):
        pnl = expiry_pnl(legs, grid, r, q)
    slope = upside_slope(legs)
    return {
        "max_profit": None if slope > 1e-9 else float(pnl.max()),
        "max_loss": None if slope < -1e-9 else float(pnl.min()),
    }


def probability_of_profit(
    legs: list[Leg], spot: float, vol: float, drift: float, r: float = 0.0, q: float = 0.0
) -> float:
    """P(P&L at the first expiry > 0) with a lognormal terminal price (volatility `vol`, drift:
    r − q for risk-neutral, the historical mean for physical — labelled by the caller)."""
    t = first_expiry(legs) / 365
    if t <= 0:
        return float(expiry_pnl(legs, np.array([spot]), r, q)[0] > 0)
    z = np.linspace(-6, 6, 4001)
    s = spot * np.exp((drift - vol**2 / 2) * t + vol * math.sqrt(t) * z)
    w = stats.norm.pdf(z)
    return float((w * (expiry_pnl(legs, s, r, q) > 0)).sum() / w.sum())


def margin_estimate(
    legs: list[Leg], spot: float, r: float = 0.0, q: float = 0.0
) -> dict[str, float | str]:
    """Reg-T-style approximation (labelled, not a broker figure). Defined-risk structures: the max
    loss. Otherwise per leg: long stock 50 %, short stock 150 %, long options the premium, naked
    short options premium + max(20 %·S − OTM amount, 10 %·S)."""
    ext = extremes(legs, spot, r, q)
    max_loss = ext["max_loss"]
    if max_loss is not None and all(leg.kind != "stock" for leg in legs):
        return {"requirement": max(-max_loss, 0.0), "basis": "defined risk: max loss"}
    req = 0.0
    for leg in legs:
        if leg.kind == "stock":
            req += (0.5 if leg.side > 0 else 1.5) * spot * leg.quantity
        elif leg.side < 0:
            covered = leg.kind == "call" and any(x.kind == "stock" and x.side > 0 for x in legs)
            if covered:
                continue
            otm = max(leg.strike - spot, 0.0) if leg.kind == "call" else max(spot - leg.strike, 0.0)
            req += leg.quantity * (leg.premium + max(0.2 * spot - otm, 0.1 * spot))
        else:
            req += leg.quantity * leg.premium
    return {
        "requirement": req,
        "basis": "Reg-T style: stock 50 % / 150 %, naked shorts 20 % / 10 %",
    }


# ── presets ──────────────────────────────────────────────────────────────────

Preset = Literal[
    "covered_call",
    "protective_put",
    "collar",
    "zero_cost_collar",
    "cash_secured_put",
    "covered_put",
    "bull_call_spread",
    "bear_call_spread",
    "bull_put_spread",
    "bear_put_spread",
    "long_straddle",
    "short_straddle",
    "long_strangle",
    "short_strangle",
    "strip",
    "strap",
    "iron_condor",
    "iron_butterfly",
    "long_call_butterfly",
    "long_put_butterfly",
    "calendar_call",
    "diagonal_call",
    "risk_reversal",
    "call_ratio_spread",
]


def priced(leg: Leg, spot: float, r: float, q: float) -> Leg:
    if leg.kind == "stock":
        return leg
    value = float(bsm_price(leg.kind, spot, leg.strike, leg.days / 365, r, q, leg.vol))
    return replace(leg, premium=value)


def build_preset(
    preset: str,
    spot: float,
    vol: float,
    days: float = 30,
    width: float = 0.05,
    wing: float = 0.05,
    r: float = 0.0,
    q: float = 0.0,
    strike_step: float = 0.0,
) -> list[Leg]:
    """Legs of a preset around the spot, priced by BSM at `vol`: `width` = distance of the inner
    strikes (fraction of S), `wing` = extra distance of the protective wings; strikes rounded to
    `strike_step` when given. Zero-cost collar: the call strike solves call = put premium."""

    def k(x: float) -> float:
        raw = spot * x
        return round(raw / strike_step) * strike_step if strike_step > 0 else raw

    def opt(kind: Kind, side: int, strike: float, quantity: float = 1.0, d: float = days) -> Leg:
        return priced(Leg(kind, side, strike, d, quantity, 0.0, vol), spot, r, q)

    stock = Leg("stock", 1, spot, days, 1.0, 0.0, vol)
    up, dn, atm = 1 + width, 1 - width, k(1.0)
    legs: list[Leg]
    if preset == "covered_call":
        legs = [stock, opt("call", -1, k(up))]
    elif preset == "protective_put":
        legs = [stock, opt("put", 1, k(dn))]
    elif preset == "collar":
        legs = [stock, opt("put", 1, k(dn)), opt("call", -1, k(up))]
    elif preset == "zero_cost_collar":
        put = opt("put", 1, k(dn))

        def gap(strike: float) -> float:
            return float(bsm_price("call", spot, strike, days / 365, r, q, vol)) - put.premium

        call_k = float(optimize.brentq(gap, spot * 0.5, spot * 10))
        legs = [stock, put, opt("call", -1, call_k)]
    elif preset == "cash_secured_put":
        legs = [opt("put", -1, k(dn))]
    elif preset == "covered_put":
        legs = [replace(stock, side=-1), opt("put", -1, k(dn))]
    elif preset == "bull_call_spread":
        legs = [opt("call", 1, atm), opt("call", -1, k(up))]
    elif preset == "bear_call_spread":
        legs = [opt("call", -1, atm), opt("call", 1, k(up))]
    elif preset == "bull_put_spread":
        legs = [opt("put", -1, atm), opt("put", 1, k(dn))]
    elif preset == "bear_put_spread":
        legs = [opt("put", 1, atm), opt("put", -1, k(dn))]
    elif preset in ("long_straddle", "short_straddle"):
        side = 1 if preset == "long_straddle" else -1
        legs = [opt("call", side, atm), opt("put", side, atm)]
    elif preset in ("long_strangle", "short_strangle"):
        side = 1 if preset == "long_strangle" else -1
        legs = [opt("call", side, k(up)), opt("put", side, k(dn))]
    elif preset == "strip":
        legs = [opt("call", 1, atm), opt("put", 1, atm, 2.0)]
    elif preset == "strap":
        legs = [opt("call", 1, atm, 2.0), opt("put", 1, atm)]
    elif preset == "iron_condor":
        legs = [
            opt("put", 1, k(dn - wing)),
            opt("put", -1, k(dn)),
            opt("call", -1, k(up)),
            opt("call", 1, k(up + wing)),
        ]
    elif preset == "iron_butterfly":
        legs = [
            opt("put", 1, k(dn)),
            opt("put", -1, atm),
            opt("call", -1, atm),
            opt("call", 1, k(up)),
        ]
    elif preset == "long_call_butterfly":
        legs = [opt("call", 1, k(dn)), opt("call", -1, atm, 2.0), opt("call", 1, k(up))]
    elif preset == "long_put_butterfly":
        legs = [opt("put", 1, k(dn)), opt("put", -1, atm, 2.0), opt("put", 1, k(up))]
    elif preset == "calendar_call":
        legs = [opt("call", -1, atm), opt("call", 1, atm, d=days * 2)]
    elif preset == "diagonal_call":
        legs = [opt("call", -1, k(up)), opt("call", 1, atm, d=days * 2)]
    elif preset == "risk_reversal":
        legs = [opt("put", -1, k(dn)), opt("call", 1, k(up))]
    elif preset == "call_ratio_spread":
        legs = [opt("call", 1, atm), opt("call", -1, k(up), 2.0)]
    else:
        raise InsufficientDataError(f"Unknown preset {preset!r}")
    return legs


# ── TR-031 systematic option-rule backtests (synthetic chains) ──────────────

Rule = Literal["covered_call_30d", "put_write_atm", "short_strangle_16d"]


def strike_for_delta(
    kind: Kind, spot: float, target: float, days: float, r: float, q: float, vol: float
) -> float:
    """Strike with |BSM Δ| = target (Brent on the strike)."""
    t = days / 365

    def gap(strike: float) -> float:
        return abs(float(bsm_greeks(kind, spot, strike, t, r, q, vol)["delta"])) - target

    return float(optimize.brentq(gap, spot * 0.2, spot * 5.0))


def rule_legs(rule: str, spot: float, vol: float, days: float, r: float, q: float) -> list[Leg]:
    def short(kind: Kind, strike: float) -> Leg:
        return priced(Leg(kind, -1, strike, days, 1.0, 0.0, vol), spot, r, q)

    if rule == "covered_call_30d":
        call_k = strike_for_delta("call", spot, 0.30, days, r, q, vol)
        return [Leg("stock", 1, spot, days, 1.0, 0.0, vol), short("call", call_k)]
    if rule == "put_write_atm":
        return [short("put", spot)]
    if rule == "short_strangle_16d":
        return [
            short("call", strike_for_delta("call", spot, 0.16, days, r, q, vol)),
            short("put", strike_for_delta("put", spot, 0.16, days, r, q, vol)),
        ]
    raise InsufficientDataError(f"Unknown rule {rule!r}")


def option_rule_backtest(
    close: pd.Series,
    iv: pd.Series,
    rule: str,
    r: float = 0.0,
    q: float = 0.0,
    days: int = 30,
    profit_take: float | None = None,
    close_dte: int | None = None,
    cost_per_option: float = 0.0,
) -> pd.DataFrame:
    """Daily mark-to-model P&L of a systematic option rule on synthetic chains: every option is
    priced by BSM at the day's IV proxy (flat smile — labelled approximation). A position opens at
    the close, is marked daily, exits at expiry or on management (option P&L ≥ `profit_take` ×
    credit, or ≤ `close_dte` days left) and is re-opened at that close. `cost_per_option` per
    option and side (an expiring option has no exit cost). Rows: pnl (per unit of underlying),
    event, interest, net = (pnl + interest) / prior close. Option-only rules are cash-secured: the
    notional (prior close) earns r on the elapsed calendar days, as in the CBOE PUT index; the
    covered call is fully invested. Each cycle is marked in one vectorised pass."""
    c = close.to_numpy(dtype=float)
    vol = iv.reindex(close.index).ffill().bfill().to_numpy(dtype=float)
    idx = pd.DatetimeIndex(close.index)
    day_no = ((idx - idx[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    n = len(c)
    pnl = np.zeros(n)
    events = [""] * n
    i = 0
    while i < n - 1:
        legs = rule_legs(rule, c[i], vol[i], days, r, q)
        n_options = sum(1 for leg in legs if leg.kind != "stock")
        credit = net_premium(legs)
        pnl[i] -= cost_per_option * n_options
        events[i] = f"{events[i]}+open" if events[i] else "open"
        last = min(int(np.searchsorted(day_no, day_no[i] + days, side="left")), n - 1)
        js = np.arange(i + 1, last + 1)
        elapsed = day_no[js] - day_no[i]
        option_marks = np.zeros(len(js))
        marks = np.zeros(len(js))
        for leg in legs:
            m = leg.pnl(c[js], elapsed, r, q, vol=vol[js])
            marks += m
            if leg.kind != "stock":
                option_marks += m
        left = days - elapsed
        exit_now = left <= 0
        if profit_take is not None and credit > 0:
            exit_now |= option_marks >= profit_take * credit
        if close_dte is not None:
            exit_now |= left <= close_dte
        hits = np.flatnonzero(exit_now)
        stop = int(hits[0]) if len(hits) else len(js) - 1
        pnl[js[: stop + 1]] += np.diff(np.concatenate([[0.0], marks[: stop + 1]]))
        e = int(js[stop])
        if len(hits):
            if left[stop] > 0:
                pnl[e] -= cost_per_option * n_options
            events[e] = "expiry" if left[stop] <= 0 else "managed"
        i = e
    frame = pd.DataFrame({"pnl": pnl, "event": events}, index=idx)
    prior = pd.Series(c, index=idx).shift(1).bfill()
    cash_secured = rule != "covered_call_30d"
    dt = pd.Series(np.diff(day_no, prepend=day_no[0]), index=idx)
    frame["interest"] = prior * r * dt / 365 if cash_secured else 0.0
    frame["net"] = (frame["pnl"] + frame["interest"]) / prior
    return frame
