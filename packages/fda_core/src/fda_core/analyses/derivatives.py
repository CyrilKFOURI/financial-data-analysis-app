"""Derivatives analytics, options as data (ANALYTICS_CATALOG DV-001…DV-012, E-006; module M21).

Inputs: `options` = {"as_of", "spot", "chain"} (latest stored snapshot), `option_history`
(daily 30-day ATM IV summaries), `vol_proxy` (VIX-family close for index ETFs), prices for
realised volatility, `company` for earnings dates. Conventions give r (short rate of the
currency) and q (trailing dividend yield).

Time to expiry T = calendar days / 365. Mid = (bid + ask)/2 when both are quoted, else last.
IVs are solved here (Brent, BSM with q), not taken from the provider; OTM options (puts below the
forward, calls above) build the smile. Measures describe the market; no trade signals (V3).
"""

import math
from datetime import date
from itertools import pairwise
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.options import (
    Heston,
    Kind,
    Svi,
    bsm_greeks,
    bsm_price,
    butterfly_g,
    calibrate_heston,
    crr_price,
    fit_svi,
    heston_price,
    implied_vol,
    implied_vol_vec,
    rnd_breeden_litzenberger,
)
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

ASSETS: set[Any] = {"asset"}
NEEDS = {"prices", "options"}
DAY = 1 / 365


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── chain preparation (pure, also used at ingestion) ─────────────────────────


def enrich_chain(chain: pd.DataFrame, spot: float, as_of: date, r: float, q: float) -> pd.DataFrame:
    """Adds T, mid, forward, log-moneyness k = ln(K/F), IV (own solver) and BSM Greeks."""
    frame = chain.copy()
    expiry = pd.to_datetime(frame["expiry"])
    frame["days"] = (expiry - pd.Timestamp(as_of)).dt.days.clip(lower=1)
    frame["t"] = frame["days"] / 365
    quoted = (frame["bid"] > 0) & (frame["ask"] > 0) & (frame["ask"] >= frame["bid"])
    frame["mid"] = np.where(quoted, (frame["bid"] + frame["ask"]) / 2, frame["last"])
    frame["spread_pct"] = np.where(quoted, (frame["ask"] - frame["bid"]) / frame["mid"], np.nan)
    frame["forward"] = spot * np.exp((r - q) * frame["t"])
    frame["k"] = np.log(frame["strike"] / frame["forward"])
    frame["iv"] = implied_vol_vec(
        (frame["kind"] == "call").to_numpy(),
        frame["mid"].fillna(0).to_numpy(dtype=float),
        spot,
        frame["strike"].to_numpy(dtype=float),
        frame["t"].to_numpy(dtype=float),
        r,
        q,
    )
    for col in ("delta", "gamma", "vega", "theta", "rho"):
        frame[col] = np.nan
    ok = frame["iv"].notna()
    for kind in ("call", "put"):
        sel = ok & (frame["kind"] == kind)
        if sel.any():
            g = bsm_greeks(
                kind,
                spot,
                frame.loc[sel, "strike"].to_numpy(),
                frame.loc[sel, "t"].to_numpy(),
                r,
                q,
                frame.loc[sel, "iv"].to_numpy(),
            )
            for col in ("delta", "gamma", "vega", "theta", "rho"):
                frame.loc[sel, col] = g[col]
    return frame


def otm(frame: pd.DataFrame) -> pd.DataFrame:
    """Out-of-the-money quotes with a solved IV (puts below the forward, calls at/above)."""
    sel = ((frame["kind"] == "put") & (frame["k"] < 0)) | (
        (frame["kind"] == "call") & (frame["k"] >= 0)
    )
    return frame.loc[sel & frame["iv"].notna() & (frame["mid"] > 0)].sort_values("k")


def fit_expiries(
    frame: pd.DataFrame, min_points: int = 6
) -> dict[date, tuple[Svi, float, pd.DataFrame]]:
    fits = {}
    for expiry, group in otm(frame).groupby("expiry"):
        g = group.loc[group["k"].abs() <= 0.6]
        if len(g) >= min_points:
            t = float(g["t"].iloc[0])
            weights = 1 / (1 + (g["spread_pct"].fillna(0.2) * 10) ** 2)
            fit, rmse = fit_svi(g["k"].to_numpy(), g["iv"].to_numpy(), t, weights.to_numpy())
            fits[pd.Timestamp(expiry).date()] = (fit, rmse, g)
    return fits


def atm_iv_at(fits: dict[date, tuple[Svi, float, pd.DataFrame]], days: float) -> float | None:
    """ATM (k = 0) IV at a horizon, linear in total variance between fitted expiries."""
    points = sorted(
        (float(g["t"].iloc[0]), float(f.w(np.array([0.0]))[0])) for f, _, g in fits.values()
    )
    if not points:
        return None
    t = days / 365
    ts, ws = [p[0] for p in points], [p[1] for p in points]
    if t <= ts[0]:
        return math.sqrt(max(ws[0], 0) / ts[0])
    if t >= ts[-1]:
        return math.sqrt(max(ws[-1], 0) / ts[-1])
    w = float(np.interp(t, ts, ws))
    return math.sqrt(max(w, 0) / t)


def chain_summary(frame: pd.DataFrame, spot: float) -> dict[str, Any]:
    fits = fit_expiries(frame)
    vol = frame.groupby("kind")["volume"].sum()
    oi = frame.groupby("kind")["open_interest"].sum()
    return {
        "atm_iv_30d": atm_iv_at(fits, 30),
        "put_call_volume": _f(vol.get("put", 0) / vol.get("call", np.nan))
        if vol.get("call", 0)
        else None,
        "put_call_oi": _f(oi.get("put", 0) / oi.get("call", np.nan)) if oi.get("call", 0) else None,
        "contracts": len(frame),
    }


# ── inputs ───────────────────────────────────────────────────────────────────


def _rates(inputs: AnalysisInputs) -> tuple[float, float]:
    return float(inputs.conventions.get("rf") or 0.0), float(
        inputs.conventions.get("dividend_yield") or 0.0
    )


def _chain(inputs: AnalysisInputs) -> tuple[date, float, pd.DataFrame]:
    data = inputs.data.get("options")
    if not isinstance(data, dict) or data.get("chain") is None or data["chain"].empty:
        raise InsufficientDataError(
            "No option chain stored: ingest options (POST /v1/options/{symbol}/ingest)"
        )
    r, q = _rates(inputs)
    key = ("_enriched", r, q)
    if key not in data:  # enrich once per request (several analyses share the snapshot)
        data[key] = enrich_chain(data["chain"], float(data["spot"]), data["as_of"], r, q)
    return data["as_of"], float(data["spot"]), data[key]


def _fits(inputs: AnalysisInputs) -> dict[date, tuple[Svi, float, pd.DataFrame]]:
    data = inputs.data["options"]
    _, _, frame = _chain(inputs)
    if "_fits" not in data:
        data["_fits"] = fit_expiries(frame)
    fits: dict[date, tuple[Svi, float, pd.DataFrame]] = data["_fits"]
    if not fits:
        raise InsufficientDataError(
            "Too few liquid quotes to fit a smile (≥ 6 OTM quotes per expiry)"
        )
    return fits


def _pick_expiry(frame: pd.DataFrame, wanted: str | None, min_days: int = 7) -> date:
    expiries: list[date] = sorted(pd.to_datetime(frame["expiry"]).dt.date.unique())
    if wanted:
        target = date.fromisoformat(wanted)
        return min(expiries, key=lambda e: abs((e - target).days))
    days = {
        e: int(frame.loc[pd.to_datetime(frame["expiry"]).dt.date == e, "days"].iloc[0])
        for e in expiries
    }
    later = [e for e in expiries if days[e] >= min_days]
    return later[0] if later else expiries[-1]


def _meta(as_of: date, spot: float, inputs: AnalysisInputs) -> dict[str, Any]:
    r, q = _rates(inputs)
    return {
        "as_of": as_of.isoformat(),
        "spot": spot,
        "r": r,
        "q": q,
        "rf_source": inputs.conventions.get("rf_source"),
    }


# ── DV-001 chain ─────────────────────────────────────────────────────────────


class ChainParams(BaseModel):
    expiry: str | None = Field(None, max_length=10, description="YYYY-MM-DD (nearest ≥ 7 days)")
    strikes: int = Field(15, ge=3, le=100, description="Strikes each side of the spot")
    min_open_interest: float = Field(0, ge=0)


@analysis(
    id="DV-001",
    version="1.0.0",
    title="Option chain",
    description=(
        "Calls | strike | puts for one expiry: bid, ask, mid, last, own IV (Brent on the mid, "
        "BSM with the currency short rate and trailing dividend yield), Δ Γ vega Θ, volume, "
        "open interest, moneyness; ±N strikes around the spot; snapshot date and quote age."
    ),
    applies_to=ASSETS,
    params=ChainParams,
    result_kind="table",
    requires=NEEDS,
)
def option_chain(inputs: AnalysisInputs, params: ChainParams) -> AnalysisOutput:
    as_of, spot, frame = _chain(inputs)
    expiry = _pick_expiry(frame, params.expiry)
    one = frame.loc[pd.to_datetime(frame["expiry"]).dt.date == expiry]
    one = one.loc[one["open_interest"].fillna(0) >= params.min_open_interest]
    strikes = sorted(one["strike"].unique())
    centre = int(np.argmin([abs(s - spot) for s in strikes])) if strikes else 0
    keep = strikes[max(0, centre - params.strikes) : centre + params.strikes + 1]
    rows = []
    for strike in keep:
        row: dict[str, Any] = {"strike": strike, "moneyness": strike / spot}
        for kind in ("call", "put"):
            hit = one.loc[(one["strike"] == strike) & (one["kind"] == kind)]
            if hit.empty:
                continue
            h = hit.iloc[0]
            prefix = kind[0]
            for col in (
                "bid",
                "ask",
                "mid",
                "iv",
                "delta",
                "gamma",
                "vega",
                "theta",
                "volume",
                "open_interest",
            ):
                row[f"{prefix}_{col}"] = _f(h[col])
        rows.append(row)
    last_trade = pd.to_datetime(one["last_trade"], utc=True).max() if "last_trade" in one else None
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "expiry": expiry.isoformat(),
            "days": int(one["days"].iloc[0]) if len(one) else None,
            "expiries": [
                e.isoformat() for e in sorted(pd.to_datetime(frame["expiry"]).dt.date.unique())
            ],
            "last_trade": None
            if last_trade is None or pd.isna(last_trade)
            else last_trade.isoformat(),
            "rows": rows,
        },
    )


# ── DV-002 pricer ────────────────────────────────────────────────────────────


class PricerParams(BaseModel):
    kind: Literal["call", "put"] = "call"
    model: Literal["bsm", "black76", "crr"] = "bsm"
    spot: float = Field(100.0, gt=0, description="Spot (or futures price for Black-76)")
    strike: float = Field(100.0, gt=0)
    days: float = Field(30.0, gt=0, le=3650)
    rate: float = Field(0.04, ge=-0.05, le=0.5)
    dividend_yield: float = Field(0.0, ge=0, le=0.5)
    volatility: float = Field(0.2, gt=0, le=5)
    steps: int = Field(500, ge=10, le=5000)
    american: bool = True
    market_price: float | None = Field(None, gt=0, description="Solve the IV from this price")


@analysis(
    id="DV-002",
    version="1.0.0",
    title="Option pricer & Greeks",
    description=(
        "Black–Scholes–Merton (continuous dividend yield), Black-76 (futures: BSM with q = r) or "
        "CRR binomial (American, N steps); Δ Γ vega Θ rho, vanna, volga, charm (BSM, per 1.00 vol "
        "and per year); IV from a market price (Brent); P&L grid over spot ±20 % × vol ±10 pts."
    ),
    applies_to=ASSETS,
    params=PricerParams,
    result_kind="table",
    requires=set(),
)
def pricer(inputs: AnalysisInputs, params: PricerParams) -> AnalysisOutput:
    kind: Kind = params.kind
    t, r = params.days / 365, params.rate
    q = r if params.model == "black76" else params.dividend_yield
    sigma = params.volatility
    iv = None
    if params.market_price is not None:
        iv = implied_vol(kind, params.market_price, params.spot, params.strike, t, r, q)
        if iv is None:
            raise InsufficientDataError("The market price is outside the no-arbitrage bounds")
        sigma = iv

    def price(s: float, vol: float) -> float:
        if params.model == "crr":
            return crr_price(kind, s, params.strike, t, r, q, vol, params.steps, params.american)
        return float(bsm_price(kind, s, params.strike, t, r, q, vol))

    value = price(params.spot, sigma)
    greeks = {
        k: float(v) for k, v in bsm_greeks(kind, params.spot, params.strike, t, r, q, sigma).items()
    }
    shifts = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    vols = [max(sigma + dv, 0.01) for dv in (-0.1, -0.05, 0.0, 0.05, 0.1)]
    grid = [[price(params.spot * (1 + ds), v) - value for ds in shifts] for v in vols]
    return AnalysisOutput(
        "table",
        {
            "model": params.model,
            "price": value,
            "implied_volatility": iv,
            "volatility_used": sigma,
            "european_bsm": float(bsm_price(kind, params.spot, params.strike, t, r, q, sigma)),
            "greeks": greeks,
            "vega_per_vol_point": greeks["vega"] / 100,
            "theta_per_day": greeks["theta"] / 365,
            "grid": {
                "rows": [f"σ {v:.0%}" for v in vols],
                "columns": [f"{s:+.0%}" for s in shifts],
                "values": grid,
                "unit": "number",
                "best": [],
            },
        },
    )


# ── DV-003 smile & skew / DV-004 term structure / DV-005 surface ─────────────


def _delta_iv(
    fit: Svi, t: float, r: float, q: float, spot: float, forward: float, target: float, kind: Kind
) -> float | None:
    """IV at a target BSM delta on the fitted smile (bisection over log-moneyness)."""
    lo, hi = -1.5, 1.5
    for _ in range(80):
        mid = (lo + hi) / 2
        strike = forward * math.exp(mid)
        iv = float(fit.iv(np.array([mid]), t)[0])
        delta = float(bsm_greeks(kind, spot, strike, t, r, q, iv)["delta"])
        if delta > target:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    return float(fit.iv(np.array([k]), t)[0])


class SmileParams(BaseModel):
    expiry: str | None = Field(None, max_length=10)


@analysis(
    id="DV-003",
    version="1.0.0",
    title="Smile & skew",
    description=(
        "Per expiry: raw SVI fit of OTM IVs on log-moneyness (weights fall with the bid/ask "
        "spread) with IV RMSE; ATM IV, 25Δ risk reversal (IV call 25Δ − IV put −25Δ), 25Δ "
        "butterfly ((call + put)/2 − ATM), ATM skew slope ∂σ/∂k; observed vs fitted smile for "
        "one expiry."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS,
)
def smile_skew(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, frame = _chain(inputs)
    fits = _fits(inputs)
    r, q = _rates(inputs)
    rows = []
    for expiry, (fit, rmse, g) in sorted(fits.items()):
        t = float(g["t"].iloc[0])
        forward = float(g["forward"].iloc[0])
        atm = float(fit.iv(np.array([0.0]), t)[0])
        c25 = _delta_iv(fit, t, r, q, spot, forward, 0.25, "call")
        p25 = _delta_iv(fit, t, r, q, spot, forward, -0.25, "put")
        slope = float(fit.dw(np.array([0.0]))[0] / (2 * atm * t))
        rows.append(
            {
                "expiry": expiry.isoformat(),
                "days": int(g["days"].iloc[0]),
                "atm_iv": atm,
                "rr_25d": None if c25 is None or p25 is None else c25 - p25,
                "bf_25d": None if c25 is None or p25 is None else (c25 + p25) / 2 - atm,
                "skew_slope": slope,
                "svi_rmse": rmse,
                "quotes": len(g),
            }
        )
    chosen = _pick_expiry(frame, params.expiry)
    fit_c = fits.get(chosen) or next(iter(sorted(fits.items())))[1]
    fit, rmse, g = fit_c
    t = float(g["t"].iloc[0])
    grid = np.linspace(float(g["k"].min()), float(g["k"].max()), 41)
    smile = [
        {"k": float(k), "strike": float(g["forward"].iloc[0] * math.exp(k)), "fitted_iv": float(v)}
        for k, v in zip(grid, fit.iv(grid, t), strict=True)
    ]
    observed = [
        {"k": float(k), "strike": float(s), "iv": float(v), "kind": kind}
        for k, s, v, kind in zip(g["k"], g["strike"], g["iv"], g["kind"], strict=True)
    ]
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "rows": rows,
            "smile_expiry": pd.Timestamp(g["expiry"].iloc[0]).date().isoformat(),
            "smile_rmse": rmse,
            "smile": smile,
            "observed": observed,
            "svi": {"a": fit.a, "b": fit.b, "rho": fit.rho, "m": fit.m, "s": fit.s},
        },
    )


def _next_earnings(inputs: AnalysisInputs, as_of: date) -> date | None:
    from fda_core.analyses.peers import _earnings

    try:
        frame = _earnings(inputs)
    except InsufficientDataError:
        return None
    upcoming = frame.loc[frame["date"].dt.date > as_of, "date"]
    return upcoming.iloc[0].date() if len(upcoming) else None


@analysis(
    id="DV-004",
    version="1.0.0",
    title="IV term structure",
    description=(
        "ATM IV (SVI at k = 0) by expiry with the forward variance between expiries; expiries "
        "spanning the next earnings date are flagged (event bump); contango/backwardation of "
        "30 vs 90 days."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS | {"company"},
)
def term_structure(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, _ = _chain(inputs)
    fits = _fits(inputs)
    earnings = _next_earnings(inputs, as_of)
    rows, previous = [], None
    for expiry, (fit, _, g) in sorted(fits.items()):
        t = float(g["t"].iloc[0])
        w = float(fit.w(np.array([0.0]))[0])
        forward_vol = None
        if previous is not None and t > previous[0] and w > previous[1]:
            forward_vol = math.sqrt((w - previous[1]) / (t - previous[0]))
        rows.append(
            {
                "expiry": expiry.isoformat(),
                "days": int(g["days"].iloc[0]),
                "atm_iv": math.sqrt(max(w, 0) / t),
                "forward_iv": forward_vol,
                "spans_earnings": bool(earnings and expiry >= earnings),
            }
        )
        previous = (t, w)
    iv30, iv90 = atm_iv_at(fits, 30), atm_iv_at(fits, 90)
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "rows": rows,
            "next_earnings": earnings.isoformat() if earnings else None,
            "atm_iv_30d": iv30,
            "atm_iv_90d": iv90,
            "shape": None
            if iv30 is None or iv90 is None
            else ("contango" if iv90 > iv30 else "backwardation"),
        },
    )


MONEYNESS = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
MIN_ARB_DAYS = 7  # shorter smiles are too noisy for meaningful arbitrage flags


@analysis(
    id="DV-005",
    version="1.0.0",
    title="IV surface",
    description=(
        "SVI-smoothed IV on a K/F × expiry grid (heatmap); arbitrage checks on expiries ≥ 7 "
        "days, within the quoted log-moneyness range: butterfly (Durrleman g(k) < 0) and "
        "calendar (total variance decreasing with maturity at fixed k); fit RMSE per expiry."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS,
)
def iv_surface(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, _ = _chain(inputs)
    fits = sorted(_fits(inputs).items())
    ks = np.log(np.array(MONEYNESS))
    values, butterfly, rmse = [], [], []
    for expiry, (fit, err, g) in fits:
        t = float(g["t"].iloc[0])
        values.append([float(v) for v in fit.iv(ks, t)])
        if int(g["days"].iloc[0]) >= MIN_ARB_DAYS:
            lo, hi = max(float(g["k"].min()), -0.5), min(float(g["k"].max()), 0.5)
            g_k = butterfly_g(fit, np.linspace(lo, hi, 101))
            if (g_k < 0).any():
                butterfly.append({"expiry": expiry.isoformat(), "min_g": float(g_k.min())})
        rmse.append(err)
    calendar = []
    checked = [f for f in fits if int(f[1][2]["days"].iloc[0]) >= MIN_ARB_DAYS]
    for (e1, (f1, _, g1)), (e2, (f2, _, g2)) in pairwise(checked):
        lo = max(float(g1["k"].min()), float(g2["k"].min()), -0.3)
        hi = min(float(g1["k"].max()), float(g2["k"].max()), 0.3)
        if hi <= lo:
            continue
        grid_k = np.linspace(lo, hi, 13)
        w1, w2 = f1.w(grid_k), f2.w(grid_k)
        bad = grid_k[w2 < w1 - 1e-6]
        if len(bad):
            calendar.append(
                {"from": e1.isoformat(), "to": e2.isoformat(), "k": [float(x) for x in bad]}
            )
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "grid": {
                "rows": [f"{e.isoformat()} ({int(g['days'].iloc[0])}d)" for e, (_, _, g) in fits],
                "columns": [f"{m:.0%}" for m in MONEYNESS],
                "values": values,
                "unit": "percent",
                "best": [],
            },
            "rmse_max": max(rmse),
            "rmse_median": float(np.median(rmse)),
            "butterfly_violations": butterfly,
            "calendar_violations": calendar,
            "arbitrage_free": not butterfly and not calendar,
        },
    )


# ── DV-006 IV rank / DV-007 cone / DV-008 VRP ───────────────────────────────


def _realised(inputs: AnalysisInputs) -> pd.Series:
    prices = inputs.data.get("prices")
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise InsufficientDataError("No stored prices for realised volatility")
    close = prices[str(inputs.conventions.get("price_field", "adj_close"))].astype(float)
    return np.log(close).diff().dropna()


def _implied_history(inputs: AnalysisInputs) -> tuple[pd.Series, str]:
    """Stored 30-day ATM IV snapshots (decimal) when ≥ 60 exist, else the VIX-family proxy for
    index products (percent → decimal), else empty."""
    hist = inputs.data.get("option_history")
    if isinstance(hist, pd.DataFrame) and hist["atm_iv_30d"].notna().sum() >= 60:
        s = hist.set_index(pd.to_datetime(hist["date"]))["atm_iv_30d"].astype(float).dropna()
        return s, "stored 30-day ATM IV snapshots"
    proxy = inputs.data.get("vol_proxy")
    if isinstance(proxy, pd.Series) and len(proxy) > 60:
        return proxy.astype(
            float
        ) / 100, f"{inputs.conventions.get('vol_proxy_symbol')} (proxy: own IV history too short)"
    return pd.Series(dtype=float), "none"


@analysis(
    id="DV-006",
    version="1.0.0",
    title="IV rank & percentile",
    description=(
        "Current 30-day ATM IV vs its 52-week history: rank (IV − min)/(max − min) and "
        "percentile (share of days below). History = stored daily snapshots (≥ 60 days), else "
        "the VIX-family proxy for index ETFs (labelled), else realised-vol rank (labelled). IV − "
        "HV(21d) spread."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS | {"option_history", "vol_proxy"},
)
def iv_rank(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, _ = _chain(inputs)
    iv_now = atm_iv_at(_fits(inputs), 30)
    r = _realised(inputs)
    hv21 = float(r.iloc[-21:].std() * math.sqrt(252))
    history, source = _implied_history(inputs)
    if history.empty:
        history = (r.rolling(21).std() * math.sqrt(252)).dropna()
        source = "realised 21-day volatility (no implied history yet)"
        current: float | None = hv21
    else:
        current = iv_now if "stored" in source else float(history.iloc[-1])
    year = history.loc[history.index >= history.index[-1] - pd.Timedelta(days=365)]
    lo, hi = float(year.min()), float(year.max())
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "atm_iv_30d": iv_now,
            "hv_21d": hv21,
            "iv_minus_hv": None if iv_now is None else iv_now - hv21,
            "history_source": source,
            "history_days": len(year),
            "rank_value": current,
            "iv_rank": (current - lo) / (hi - lo) if hi > lo and current is not None else None,
            "iv_percentile": float((year < current).mean()) if current is not None else None,
            "low_52w": lo,
            "high_52w": hi,
        },
    )


HORIZONS = (10, 21, 42, 63, 126, 252)


@analysis(
    id="DV-007",
    version="1.0.0",
    title="Volatility cone",
    description=(
        "Realised volatility (√252) over rolling windows of 10…252 sessions on the stored "
        "history: min, p10, p25, median, p75, p90, max and the latest value per horizon, with the "
        "current ATM IV term structure at the same horizons (calendar days ≈ sessions × 365/252)."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS,
)
def vol_cone(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, _ = _chain(inputs)
    r = _realised(inputs)
    try:
        fits = _fits(inputs)
    except InsufficientDataError:
        fits = {}
    rows = []
    for h in HORIZONS:
        rv = (r.rolling(h).std() * math.sqrt(252)).dropna()
        if len(rv) < 20:
            continue
        rows.append(
            {
                "horizon": h,
                **{
                    f"p{int(p * 100)}": float(rv.quantile(p))
                    for p in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
                },
                "current_rv": float(rv.iloc[-1]),
                "implied": atm_iv_at(fits, h * 365 / 252) if fits else None,
            }
        )
    if not rows:
        raise InsufficientDataError("Volatility cone needs ≥ 30 sessions of prices")
    return AnalysisOutput("table", {**_meta(as_of, spot, inputs), "rows": rows, "sessions": len(r)})


class VrpParams(BaseModel):
    horizon: int = Field(21, ge=5, le=63)


@analysis(
    id="DV-008",
    version="1.0.0",
    title="Volatility risk premium",
    description=(
        "Implied 30-day volatility at t minus the realised volatility of the next `horizon` "
        "sessions (ex post, so the last horizon days have no value); mean, share positive. "
        "Implied history as DV-006 (own snapshots or VIX-family proxy)."
    ),
    applies_to=ASSETS,
    params=VrpParams,
    result_kind="series",
    requires={"prices", "option_history", "vol_proxy"},
)
def vrp(inputs: AnalysisInputs, params: VrpParams) -> AnalysisOutput:
    implied, source = _implied_history(inputs)
    if implied.empty:
        raise InsufficientDataError(
            "No implied-volatility history (own snapshots < 60 days, no VIX-family proxy)"
        )
    r = _realised(inputs)
    future_rv = (r.rolling(params.horizon).std() * math.sqrt(252)).shift(-params.horizon)
    both = pd.concat([implied.rename("iv"), future_rv.rename("rv")], axis=1, join="inner").dropna()
    if len(both) < 60:
        raise InsufficientDataError(f"VRP needs ≥ 60 overlapping days, got {len(both)}")
    spread = both["iv"] - both["rv"]
    shown = spread.iloc[-756:]
    return AnalysisOutput(
        "series",
        {
            "source": source,
            "horizon": params.horizon,
            "mean_vrp": float(spread.mean()),
            "share_positive": float((spread > 0).mean()),
            "observations": len(spread),
            "dates": [d.date().isoformat() for d in pd.to_datetime(shown.index)],
            "lines": {
                "vrp": [float(v) for v in shown],
                "implied": [float(v) for v in both["iv"].iloc[-756:]],
                "realised_next": [float(v) for v in both["rv"].iloc[-756:]],
            },
        },
    )


# ── DV-009 positioning ───────────────────────────────────────────────────────


class PositioningParams(BaseModel):
    expiry: str | None = Field(None, max_length=10, description="Default: all expiries")
    unusual_ratio: float = Field(2.0, gt=0)
    unusual_multiple: float = Field(10.0, gt=0)


def max_pain(frame: pd.DataFrame) -> float | None:
    """Strike minimising the intrinsic value paid to option holders at expiry (descriptive)."""
    strikes = np.sort(frame["strike"].unique())
    if not len(strikes):
        return None
    oi = frame.assign(open_interest=frame["open_interest"].fillna(0))  # unreported OI = none
    calls = oi.loc[oi["kind"] == "call", ["strike", "open_interest"]].to_numpy(dtype=float)
    puts = oi.loc[oi["kind"] == "put", ["strike", "open_interest"]].to_numpy(dtype=float)
    pain = [
        float(
            (np.maximum(s - calls[:, 0], 0) * calls[:, 1]).sum()
            + (np.maximum(puts[:, 0] - s, 0) * puts[:, 1]).sum()
        )
        for s in strikes
    ]
    return float(strikes[int(np.argmin(pain))])


@analysis(
    id="DV-009",
    version="1.0.0",
    title="Positioning",
    description=(
        "Open interest and volume by strike, put/call ratios (volume, open interest), max pain "
        "for one expiry (default: nearest ≥ 7 days; strike minimising holders' intrinsic value, "
        "unreported OI = 0, descriptive), unusual activity: volume / "
        "OI > ratio and volume > multiple × median traded-contract volume (one snapshot: no "
        "volume history)."
    ),
    applies_to=ASSETS,
    params=PositioningParams,
    result_kind="table",
    requires=NEEDS,
)
def positioning(inputs: AnalysisInputs, params: PositioningParams) -> AnalysisOutput:
    as_of, spot, frame = _chain(inputs)
    scope = frame
    pain_expiry = _pick_expiry(frame, params.expiry)
    if params.expiry:
        scope = frame.loc[pd.to_datetime(frame["expiry"]).dt.date == pain_expiry]
    by_strike = scope.pivot_table(
        index="strike", columns="kind", values=["open_interest", "volume"], aggfunc="sum"
    ).fillna(0)
    rows = [
        {
            "strike": float(k),
            "call_oi": float(v.get(("open_interest", "call"), 0)),
            "put_oi": float(v.get(("open_interest", "put"), 0)),
            "call_volume": float(v.get(("volume", "call"), 0)),
            "put_volume": float(v.get(("volume", "put"), 0)),
        }
        for k, v in by_strike.iterrows()
        if abs(float(k) / spot - 1) <= 0.3
    ]
    vol = scope.groupby("kind")["volume"].sum()
    oi = scope.groupby("kind")["open_interest"].sum()
    traded = scope.loc[scope["volume"] > 0, "volume"]
    median_volume = float(traded.median()) if len(traded) else 0.0
    unusual = scope.loc[
        (scope["volume"] > params.unusual_ratio * scope["open_interest"].clip(lower=1))
        & (scope["volume"] > params.unusual_multiple * max(median_volume, 1))
    ]
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "put_call_volume": _f(vol.get("put", 0) / vol.get("call")) if vol.get("call") else None,
            "put_call_oi": _f(oi.get("put", 0) / oi.get("call")) if oi.get("call") else None,
            "max_pain": max_pain(frame.loc[pd.to_datetime(frame["expiry"]).dt.date == pain_expiry]),
            "max_pain_expiry": pain_expiry.isoformat(),
            "rows": rows,
            "unusual": [
                {
                    "contract": c,
                    "expiry": pd.Timestamp(e).date().isoformat(),
                    "kind": kind,
                    "strike": float(s),
                    "volume": float(v),
                    "open_interest": float(o),
                }
                for c, e, kind, s, v, o in unusual[
                    ["contract", "expiry", "kind", "strike", "volume", "open_interest"]
                ]
                .head(30)
                .itertuples(index=False)
            ],
        },
    )


# ── DV-010 expected move / E-006 ─────────────────────────────────────────────


def straddle_moves(frame: pd.DataFrame, spot: float) -> list[dict[str, Any]]:
    out = []
    for expiry, group in frame.groupby("expiry"):
        strikes = group["strike"].unique()
        atm = float(strikes[np.argmin(np.abs(strikes - spot))])
        legs = group.loc[group["strike"] == atm].set_index("kind")["mid"]
        if {"call", "put"} <= set(legs.index) and legs.notna().all():
            straddle = float(legs["call"] + legs["put"])
            iv = group.loc[group["strike"] == atm, "iv"].mean()
            t = float(group["t"].iloc[0])
            out.append(
                {
                    "expiry": pd.Timestamp(expiry).date().isoformat(),
                    "days": int(group["days"].iloc[0]),
                    "atm_strike": atm,
                    "straddle": straddle,
                    "move_pct": straddle / spot,
                    "one_sigma_pct": _f(iv * math.sqrt(t)) if pd.notna(iv) else None,
                }
            )
    return out


def earnings_moves(inputs: AnalysisInputs, as_of: date) -> list[dict[str, Any]]:
    """Absolute close-to-close move on the first session after each past report (reports
    after the close move the next session; the larger of the report-day and next-day move)."""
    from fda_core.analyses.peers import _earnings

    try:
        dates = _earnings(inputs)["date"]
    except InsufficientDataError:
        return []
    prices = inputs.data.get("prices")
    if not isinstance(prices, pd.DataFrame):
        return []
    close = prices[str(inputs.conventions.get("price_field", "adj_close"))].astype(float)
    r = close.pct_change()
    out = []
    for d in dates:
        day = d.tz_convert(None).normalize() if d.tzinfo else d.normalize()
        if day.date() >= as_of:
            continue
        idx = r.index.searchsorted(day)
        window = r.iloc[idx : idx + 2]
        if len(window):
            move = float(window.abs().max())
            out.append({"date": day.date().isoformat(), "abs_move": move})
    return out[-12:]


class MoveParams(BaseModel):
    pass


def _expected_move(inputs: AnalysisInputs) -> dict[str, Any]:
    as_of, spot, frame = _chain(inputs)
    moves = straddle_moves(frame, spot)
    earnings = _next_earnings(inputs, as_of)
    history = earnings_moves(inputs, as_of)
    implied_earnings = None
    if earnings:
        after = [m for m in moves if date.fromisoformat(m["expiry"]) >= earnings]
        if after:
            implied_earnings = after[0]
    avg = float(np.mean([h["abs_move"] for h in history])) if history else None
    return {
        **_meta(as_of, spot, inputs),
        "rows": moves,
        "next_earnings": earnings.isoformat() if earnings else None,
        "earnings_expiry": implied_earnings,
        "implied_earnings_move_pct": implied_earnings["move_pct"] if implied_earnings else None,
        "historical_avg_abs_move": avg,
        "historical_moves": history,
    }


@analysis(
    id="DV-010",
    version="1.0.0",
    title="Expected move",
    description=(
        "Per expiry: ATM straddle mid as the expected ± move (≈ 0.8 σ√T S) and the 1σ move "
        "σ√T; for the first expiry after the next earnings date, the implied move next to the "
        "average absolute move after the last 12 reports."
    ),
    applies_to=ASSETS,
    params=MoveParams,
    result_kind="table",
    requires=NEEDS | {"company"},
)
def expected_move(inputs: AnalysisInputs, params: MoveParams) -> AnalysisOutput:
    return AnalysisOutput("table", _expected_move(inputs))


@analysis(
    id="E-006",
    version="1.0.0",
    title="Implied vs realised earnings move",
    description=(
        "Straddle-implied move to the first expiry after the next report vs the average absolute "
        "one-session move after the last 12 reports (DV-010, earnings view)."
    ),
    applies_to=ASSETS,
    params=MoveParams,
    result_kind="table",
    requires=NEEDS | {"company"},
)
def earnings_implied_move(inputs: AnalysisInputs, params: MoveParams) -> AnalysisOutput:
    out = _expected_move(inputs)
    if out["next_earnings"] is None:
        raise InsufficientDataError("No upcoming earnings date (ingest company data)")
    keep = (
        "as_of",
        "spot",
        "next_earnings",
        "earnings_expiry",
        "implied_earnings_move_pct",
        "historical_avg_abs_move",
        "historical_moves",
    )
    return AnalysisOutput("table", {k: out[k] for k in keep})


# ── DV-011 risk-neutral density / DV-012 Heston ──────────────────────────────


@analysis(
    id="DV-011",
    version="1.0.0",
    title="Risk-neutral density",
    description=(
        "Breeden–Litzenberger f(K) = e^{rT} ∂²C/∂K² on the SVI-smoothed smile of one expiry "
        "(calls priced with Black on the forward); mean, mode, 5/50/95 % quantiles, probability "
        "below −10 % and above +10 % of the spot. Risk-neutral, not a forecast."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS,
)
def risk_neutral_density(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, frame = _chain(inputs)
    fits = _fits(inputs)
    r, _ = _rates(inputs)
    expiry = _pick_expiry(frame, params.expiry, min_days=14)
    if expiry not in fits:
        expiry = min(fits, key=lambda e: abs((e - expiry).days))
    fit, rmse, g = fits[expiry]
    t, forward = float(g["t"].iloc[0]), float(g["forward"].iloc[0])
    strikes = np.linspace(forward * 0.2, forward * 2.5, 2000)
    density = rnd_breeden_litzenberger(fit, forward, t, r, strikes)
    mass = float(np.trapezoid(density, strikes))
    if mass <= 0:
        raise InsufficientDataError("Degenerate density from the fitted smile")
    cdf = np.cumsum(density) * (strikes[1] - strikes[0]) / mass

    def q(p: float) -> float:
        return float(strikes[min(int(np.searchsorted(cdf, p)), len(strikes) - 1)])

    step = max(len(strikes) // 200, 1)
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "expiry": expiry.isoformat(),
            "days": int(g["days"].iloc[0]),
            "svi_rmse": rmse,
            "mass": mass,
            "mean": float(np.trapezoid(density * strikes, strikes) / mass),
            "mode": float(strikes[int(np.argmax(density))]),
            "q05": q(0.05),
            "q50": q(0.5),
            "q95": q(0.95),
            "prob_below_minus_10pct": float(np.interp(spot * 0.9, strikes, cdf)),
            "prob_above_plus_10pct": float(1 - np.interp(spot * 1.1, strikes, cdf)),
            "density": [
                {"strike": float(k), "density": float(d)}
                for k, d in zip(strikes[::step], density[::step] / mass, strict=True)
            ],
        },
    )


@analysis(
    id="DV-012",
    version="1.0.0",
    title="Heston calibration",
    description=(
        "Heston (κ, θ, σ, rho, v₀) fitted by least squares to OTM mid prices (up to 4 expiries "
        "≥ 14 days, |k| ≤ 0.25, ≤ 60 quotes); price and IV RMSE, Feller 2κθ > σ² reported. An "
        "analysis of the surface, not a trading model."
    ),
    applies_to=ASSETS,
    params=SmileParams,
    result_kind="table",
    requires=NEEDS,
    budget_ms=20_000,
)
def heston_calibration(inputs: AnalysisInputs, params: SmileParams) -> AnalysisOutput:
    as_of, spot, frame = _chain(inputs)
    r, q = _rates(inputs)
    pool = otm(frame)
    pool = pool.loc[(pool["days"] >= 14) & (pool["k"].abs() <= 0.25)]
    expiries = sorted(pool["expiry"].unique())[:4]
    pool = pool.loc[pool["expiry"].isin(expiries)]
    if len(pool) < 10:
        raise InsufficientDataError(f"Heston needs ≥ 10 OTM quotes ≥ 14 days out, got {len(pool)}")
    pool = pool.groupby("expiry", group_keys=False).apply(lambda g: g.iloc[:: max(len(g) // 15, 1)])
    quotes: list[tuple[Kind, float, float, float]] = [
        ("call" if kind == "call" else "put", float(k), float(t), float(m))
        for kind, k, t, m in pool[["kind", "strike", "t", "mid"]].itertuples(index=False)
    ]
    atm = atm_iv_at(fit_expiries(frame), 30) or 0.2
    fit, rmse = calibrate_heston(spot, r, q, quotes, Heston(2.0, atm**2, 0.5, -0.6, atm**2))
    model = np.array(
        [
            float(heston_price(kind, spot, np.array([k]), t, r, q, fit)[0])
            for kind, k, t, _ in quotes
        ]
    )
    model_iv = [
        implied_vol(kind, p, spot, k, t, r, q)
        for (kind, k, t, _), p in zip(quotes, model, strict=True)
    ]
    iv_err = [mi - oi for mi, oi in zip(model_iv, pool["iv"], strict=True) if mi is not None]
    return AnalysisOutput(
        "table",
        {
            **_meta(as_of, spot, inputs),
            "kappa": fit.kappa,
            "theta": fit.theta,
            "sigma": fit.sigma,
            "rho": fit.rho,
            "v0": fit.v0,
            "long_run_vol": math.sqrt(fit.theta),
            "spot_vol": math.sqrt(fit.v0),
            "feller": 2 * fit.kappa * fit.theta > fit.sigma**2,
            "price_rmse": rmse,
            "at_bounds": [
                n
                for n, v, lo, hi in (
                    ("kappa", fit.kappa, 0.05, 20.0),
                    ("theta", fit.theta, 1e-4, 1.0),
                    ("sigma", fit.sigma, 0.01, 3.0),
                    ("rho", fit.rho, -0.99, 0.99),
                    ("v0", fit.v0, 1e-4, 1.0),
                )
                if min(abs(v - lo), abs(v - hi)) < 1e-3 * (hi - lo)  # within 0.1 % of the range
            ],
            "iv_rmse": float(np.sqrt(np.mean(np.square(iv_err)))) if iv_err else None,
            "quotes": len(quotes),
            "expiries": [pd.Timestamp(e).date().isoformat() for e in expiries],
        },
    )
