"""Investment plans: DCA, rebalancing, hedging, risk budgets, goals, Kelly (ANALYTICS_CATALOG
IN-020…IN-025; module M24). Model outputs, labelled; nothing is executed.

DCA rule tables follow M24 §2 (Quantrick ch.4 presets, to be re-checked after R-QT):
VIX multiplier < 15 → 0.5×, 15–20 → 1×, 20–30 → 1.5×, > 30 → 2×; trend filter pauses below the
200-day average and doubles the next contribution on re-cross; hybrid = base + 50 % when VIX > 25,
pause when below the average and VIX < 20, value-averaging top-up (≤ 1 base). Decisions at each
contribution date use only data known on that date (no look-ahead).
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from fda_core.analyses.invest import (
    MODEL_ALLOCATIONS,
    MODEL_LABEL,
    CmaParams,
    _profile,
    capital_market_assumptions,
    classify_holding,
    glide_score,
    score_profile,
)
from fda_core.analyses.portfolio import _assets, _current, _pf
from fda_core.analyses.price import PRICE_SUBJECTS, _field, _prices
from fda_core.analyses.risk import STRESS_WINDOWS
from fda_core.options import bsm_price
from fda_core.portfolio import risk_contributions
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

Variant = Literal["lump_sum", "dca", "value_averaging", "vix_scaled", "trend_filtered", "hybrid"]
VARIANTS: tuple[Variant, ...] = (
    "lump_sum",
    "dca",
    "value_averaging",
    "vix_scaled",
    "trend_filtered",
    "hybrid",
)


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── IN-020 DCA rules (pure, unit-tested) ─────────────────────────────────────


def vix_multiplier(vix: float | None) -> float:
    if vix is None or not math.isfinite(vix):
        return 1.0
    if vix < 15:
        return 0.5
    if vix <= 20:
        return 1.0
    if vix <= 30:
        return 1.5
    return 2.0


def simulate_plan(
    variant: Variant,
    prices: np.ndarray,
    sma: np.ndarray,
    vix: np.ndarray,
    base: float,
    cash_yield_monthly: float,
    va_growth: float = 0.01,
    allow_selling: bool = False,
) -> dict[str, float]:
    """One cohort, monthly steps t = 0…T−1 at month-end prices; contributions invest at that
    month's price; paused money waits in cash (yielding) and is invested per the rule; the
    lump sum invests the whole planned budget (T × base) at t = 0. Terminal wealth = units ×
    last price + cash. Returns wealth, invested, contributed, IRR, max drawdown of wealth."""
    months = len(prices)
    budget = base * months
    units, cash = 0.0, 0.0
    flows: list[float] = []  # money put into the plan each month (negative = withdrawn)
    wealth_path = []
    pending_double = False
    for t in range(months):
        price, below, v = prices[t], prices[t] < sma[t], vix[t]
        cash *= 1 + cash_yield_monthly
        if variant == "lump_sum":
            amount = budget if t == 0 else 0.0
            new_money = amount
        elif variant == "dca":
            amount = new_money = base
        elif variant == "vix_scaled":
            amount = new_money = base * vix_multiplier(v)
        elif variant == "trend_filtered":
            new_money = base
            if below:
                cash += base
                amount = 0.0
                pending_double = True
            else:
                amount = base * (2 if pending_double else 1)
                pending_double = False
                take = min(cash, amount - base)  # the extra comes from paused cash
                cash -= take
                amount = base + take
        elif variant == "value_averaging":
            target = base * sum((1 + va_growth) ** k for k in range(t + 1))
            gap = target - units * price
            amount = gap if (gap > 0 or allow_selling) else 0.0
            new_money = base
            cash += base - amount  # surplus waits in cash; deficits drawn from cash first
        else:  # hybrid
            new_money = base
            boost = 1.5 if (v is not None and math.isfinite(v) and v > 25) else 1.0
            if below and (v is not None and math.isfinite(v) and v < 20):
                cash += base
                amount = 0.0
            else:
                target = base * sum((1 + va_growth) ** k for k in range(t + 1))
                top_up = min(max(target - units * price - base * boost, 0.0), base, cash)
                amount = base * boost + top_up
                cash += base - amount  # boost and top-up are funded by waiting cash or new money
        if variant in ("value_averaging", "hybrid") and cash < 0:
            new_money += -cash
            cash = 0.0
        flows.append(new_money)
        units += amount / price
        wealth_path.append(units * price + cash)
    terminal = units * prices[-1] + cash
    contributed = float(sum(flows))
    wealth = np.array(wealth_path)
    peak = np.maximum.accumulate(wealth)
    mdd = float(((wealth - peak) / np.where(peak > 0, peak, 1)).min())
    return {
        "wealth": terminal,
        "contributed": contributed,
        "multiple": terminal / contributed if contributed else math.nan,
        "irr": monthly_irr(flows, terminal),
        "max_drawdown": mdd,
        "cash_left": cash,
    }


def monthly_irr(flows: list[float], terminal: float) -> float:
    """Annualised IRR of monthly contributions (start of month) against the terminal value."""
    n = len(flows)

    def npv(r: float) -> float:
        m = (1 + r) ** (1 / 12) - 1
        return float(sum(f * (1 + m) ** (n - k) for k, f in enumerate(flows))) - terminal

    try:
        from scipy import optimize

        return float(optimize.brentq(npv, -0.95, 5.0))
    except ValueError:
        return math.nan


class DcaParams(BaseModel):
    monthly_amount: float = Field(1000.0, gt=0)
    months: int = Field(60, ge=12, le=360)
    cash_yield: float = Field(0.02, ge=0, le=0.2, description="Annual yield on waiting cash")
    va_growth: float = Field(0.01, ge=0, le=0.05, description="Value-averaging path growth / month")
    allow_selling: bool = False
    simulations: int = Field(1000, ge=100, le=10000)
    block_months: int = Field(12, ge=1, le=36)


def _monthly(inputs: AnalysisInputs) -> pd.DataFrame:
    prices = _prices(inputs)
    close = prices[_field(inputs)].astype(float)
    sma = close.rolling(200, min_periods=200).mean()
    vix = inputs.data.get("vix")
    frame = pd.DataFrame({"price": close, "sma": sma})
    if isinstance(vix, pd.Series):
        frame["vix"] = vix.reindex(frame.index, method="ffill")
    else:
        frame["vix"] = np.nan
    return frame.resample("ME").last().dropna(subset=["price", "sma"])


@analysis(
    id="IN-020",
    version="1.0.0",
    title="DCA planner",
    description=(
        "Lump sum vs standard DCA, value averaging, VIX-scaled DCA (0.5×/1×/1.5×/2× at <15/15–20/"
        "20–30/>30), trend-filtered (pause below the 200-day average, double on re-cross) and "
        "hybrid (+50 % when VIX > 25, pause when below the average and VIX < 20, VA top-up). "
        "Rolling monthly cohorts over the stored history (decisions on data known at each date) "
        "and a block-bootstrap Monte Carlo: terminal multiple, IRR, max drawdown, cash drag, share "
        "of cohorts beating the lump sum; the next 12 contributions under today's state."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DcaParams,
    result_kind="table",
    requires={"prices", "vix"},
    budget_ms=30_000,
)
def dca_planner(inputs: AnalysisInputs, params: DcaParams) -> AnalysisOutput:
    m = _monthly(inputs)
    if len(m) < params.months + 12:
        raise InsufficientDataError(
            f"Needs ≥ {params.months + 12} months with a 200-day average, got {len(m)}"
        )
    p, s, v = m["price"].to_numpy(), m["sma"].to_numpy(), m["vix"].to_numpy()
    cy = (1 + params.cash_yield) ** (1 / 12) - 1
    cohorts: dict[str, list[dict[str, float]]] = {k: [] for k in VARIANTS}
    for start in range(0, len(m) - params.months + 1):
        sl = slice(start, start + params.months)
        for variant in VARIANTS:
            cohorts[variant].append(
                simulate_plan(
                    variant,
                    p[sl],
                    s[sl],
                    v[sl],
                    params.monthly_amount,
                    cy,
                    params.va_growth,
                    params.allow_selling,
                )
            )
    lump = np.array([c["multiple"] for c in cohorts["lump_sum"]])
    rows: list[dict[str, Any]] = []
    for name, runs in cohorts.items():
        mult = np.array([c["multiple"] for c in runs])
        rows.append(
            {
                "variant": name,
                "cohorts": len(runs),
                "median_multiple": float(np.median(mult)),
                "p5_multiple": float(np.quantile(mult, 0.05)),
                "p95_multiple": float(np.quantile(mult, 0.95)),
                "median_irr": float(np.nanmedian([c["irr"] for c in runs])),
                "median_max_drawdown": float(np.median([c["max_drawdown"] for c in runs])),
                "avg_cash_left": float(np.mean([c["cash_left"] for c in runs])),
                "beats_lump_sum": None if name == "lump_sum" else float((mult > lump).mean()),
            }
        )
    # Monte Carlo: monthly blocks resampled jointly (returns, VIX, above/below-trend state).
    rng = np.random.default_rng(17)
    ret = np.diff(np.log(p))
    gap = np.log(p / s)
    mc_rows = []
    n_blocks = math.ceil(params.months / params.block_months)
    sims: dict[str, list[float]] = {k: [] for k in VARIANTS}
    for _ in range(params.simulations):
        idx = np.concatenate(
            [
                np.arange(b, b + params.block_months)
                for b in rng.integers(0, len(ret) - params.block_months, n_blocks)
            ]
        )[: params.months]
        path = 100 * np.exp(np.cumsum(ret[idx]))
        sma_path = path / np.exp(gap[idx + 1])  # keep the sampled month's distance to its average
        for variant in VARIANTS:
            sims[variant].append(
                simulate_plan(
                    variant,
                    path,
                    sma_path,
                    v[idx + 1],
                    params.monthly_amount,
                    cy,
                    params.va_growth,
                    params.allow_selling,
                )["multiple"]
            )
    lump_mc = np.array(sims["lump_sum"])
    for name, mults in sims.items():
        arr = np.array(mults)
        mc_rows.append(
            {
                "variant": name,
                "median_multiple": float(np.median(arr)),
                "p5_multiple": float(np.quantile(arr, 0.05)),
                "p95_multiple": float(np.quantile(arr, 0.95)),
                "beats_lump_sum": None if name == "lump_sum" else float((arr > lump_mc).mean()),
            }
        )
    last_vix = _f(v[-1])
    below_now = bool(p[-1] < s[-1])
    schedule = [
        {
            "month": k + 1,
            "dca": params.monthly_amount,
            "vix_scaled": params.monthly_amount * vix_multiplier(last_vix),
            "trend_filtered": 0.0 if below_now else params.monthly_amount,
            "hybrid": 0.0
            if below_now and last_vix is not None and last_vix < 20
            else params.monthly_amount * (1.5 if (last_vix or 0) > 25 else 1.0),
        }
        for k in range(12)
    ]
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "rules": "M24 §2 (Quantrick ch.4 presets; re-check after R-QT)",
            "months": params.months,
            "history": [m.index[0].date().isoformat(), m.index[-1].date().isoformat()],
            "vix_available": bool(np.isfinite(v).any()),
            "rows": rows,
            "monte_carlo": mc_rows,
            "state_now": {"vix": last_vix, "below_200dma": below_now},
            "schedule": schedule[:3],
        },
    )


# ── IN-021 rebalancing policy designer ───────────────────────────────────────

POLICIES: dict[str, dict[str, Any]] = {
    "none": {"type": "none"},
    "monthly": {"type": "calendar", "freq": "M"},
    "quarterly": {"type": "calendar", "freq": "Q"},
    "annual": {"type": "calendar", "freq": "A"},
    "band_5pt": {"type": "band", "band": 0.05},
    "band_20pct_rel": {"type": "relative", "band": 0.20},
    "quarterly_check_5pt": {"type": "hybrid", "freq": "Q", "band": 0.05},
}


def rebalance_sim(
    returns: pd.DataFrame, target: np.ndarray, policy: dict[str, Any], cost_bps: float
) -> dict[str, float]:
    """Daily drift; trade back to target when the policy fires; cost = bps × one-way turnover."""
    x = returns.to_numpy(dtype=float)
    w = target.copy()
    periods = returns.index.to_period(
        {"M": "M", "Q": "Q", "A": "Y"}.get(policy.get("freq", "M"), "M")
    )
    daily, turnover_total, trades = [], 0.0, 0
    for i in range(len(x)):
        fire = False
        new_period = i > 0 and periods[i] != periods[i - 1]
        drift = np.abs(w - target)
        kind = policy["type"]
        if kind == "calendar":
            fire = new_period
        elif kind == "band":
            fire = bool((drift > policy["band"]).any())
        elif kind == "relative":
            fire = bool((drift > policy["band"] * np.maximum(target, 1e-9)).any())
        elif kind == "hybrid":
            fire = new_period and bool((drift > policy["band"]).any())
        cost = 0.0
        if fire:
            t = float(np.abs(target - w).sum() / 2)
            turnover_total += t
            trades += 1
            cost = t * cost_bps / 10_000
            w = target.copy()
        r = float(w @ x[i]) - cost
        daily.append(r)
        grown = w * (1 + x[i])
        w = grown / grown.sum()
    port = np.array(daily)
    ideal = x @ target
    years = len(x) / 252
    return {
        "tracking_error": float(np.std(port - ideal, ddof=1) * math.sqrt(252)),
        "turnover_per_year": turnover_total / years,
        "rebalances_per_year": trades / years,
        "return": float(np.expm1(np.log1p(port).sum() / years)),
        "volatility": float(port.std(ddof=1) * math.sqrt(252)),
        "cost_per_year": turnover_total * cost_bps / 10_000 / years,
    }


class RebalanceParams(BaseModel):
    cost_bps: float = Field(10.0, ge=0, le=200)
    max_turnover: float = Field(0.5, gt=0, le=5, description="Annual one-way turnover budget")


@analysis(
    id="IN-021",
    version="1.0.0",
    title="Rebalancing policy designer",
    description=(
        "The portfolio's targets under no rebalancing, calendar (M/Q/A), absolute 5-pt band, "
        "relative 20 % band and quarterly check + 5-pt band: tracking error vs a daily-rebalanced "
        "target, turnover, trades and cost per year; recommended policy = lowest tracking error "
        "within the turnover budget, with IPS wording."
    ),
    applies_to={"portfolio"},
    params=RebalanceParams,
    result_kind="table",
    requires={"portfolio"},
)
def rebalancing_designer(inputs: AnalysisInputs, params: RebalanceParams) -> AnalysisOutput:
    pf = _pf(inputs)
    holdings = [h for h in (pf.definition.get("holdings") or []) if not h.get("is_cash")]
    if not holdings:
        raise InsufficientDataError("Rebalancing needs target weights (weights-mode portfolio)")
    assets = [h["symbol"] for h in holdings if h["symbol"] in pf.path.returns.columns]
    returns = pf.path.returns[assets].iloc[1:].dropna()
    target = np.array(
        [float(next(h["weight"] for h in holdings if h["symbol"] == a)) for a in assets]
    )
    target = target / target.sum()
    if len(returns) < 252:
        raise InsufficientDataError(f"Needs ≥ 252 sessions of holding returns, got {len(returns)}")
    rows: list[dict[str, Any]] = [
        {"policy": name, **rebalance_sim(returns, target, spec, params.cost_bps)}
        for name, spec in POLICIES.items()
    ]
    eligible = [
        r for r in rows if r["turnover_per_year"] <= params.max_turnover and r["policy"] != "none"
    ]
    best = min(eligible, key=lambda r: r["tracking_error"]) if eligible else None
    wording = None
    if best:
        spec = POLICIES[best["policy"]]
        period = {"M": "month", "Q": "quarter", "A": "year"}[spec.get("freq", "M")]
        band = spec.get("band", 0) * 100
        wording = {
            "calendar": f"Rebalance to target every {period}.",
            "band": f"Rebalance when any weight drifts more than {band:.0f} points from target.",
            "relative": f"Rebalance when any weight drifts more than {band:.0f} % of its target.",
            "hybrid": "Check quarterly; rebalance only weights more than 5 points from target.",
        }[spec["type"]]
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "window": [returns.index[0].date().isoformat(), returns.index[-1].date().isoformat()],
            "rows": rows,
            "recommended": best["policy"] if best else None,
            "ips_wording": wording,
            "cost_bps": params.cost_bps,
        },
    )


# ── IN-023 risk budgeting ────────────────────────────────────────────────────


class RiskBudgetParams(BaseModel):
    budgets: dict[str, float] = Field(default_factory=dict, description="sleeve → share of risk")
    target_vol: float = Field(0.12, gt=0, le=1)
    alpha: float = Field(0.95, ge=0.8, le=0.995)


@analysis(
    id="IN-023",
    version="1.0.0",
    title="Risk budgeting plan",
    description=(
        "Euler volatility and historical CVaR contributions per sleeve (asset classes of the "
        "holdings; core/satellite) vs budgets; budget usage; volatility-targeting exposure = "
        "target vol / current ex-ante vol (descriptive overlay)."
    ),
    applies_to={"portfolio"},
    params=RiskBudgetParams,
    result_kind="table",
    requires={"portfolio"},
)
def risk_budgeting(inputs: AnalysisInputs, params: RiskBudgetParams) -> AnalysisOutput:
    pf = _pf(inputs)
    weights = _current(pf)
    assets = [a for a in _assets(pf) if a in weights.index]
    r = pf.path.returns[assets].iloc[1:].dropna().iloc[-756:]
    if len(r) < 120:
        raise InsufficientDataError(f"Needs ≥ 120 sessions of holding returns, got {len(r)}")
    w = np.array([float(weights[a]) for a in assets])
    cov = np.cov(r.to_numpy(dtype=float), rowvar=False) * 252
    sigma, _, rc = risk_contributions(w, cov)
    port = r.to_numpy(dtype=float) @ w
    losses = -port
    var = np.quantile(losses, params.alpha)
    tail = losses >= var
    cvar_contrib = (
        -(r.to_numpy(dtype=float)[tail] * w).mean(axis=0) if tail.any() else np.zeros(len(w))
    )
    sleeves: dict[str, dict[str, float]] = {}
    for i, a in enumerate(assets):
        meta = pf.meta.get(a, {})
        key = classify_holding(a, meta.get("asset_class"), meta.get("country"))
        s = sleeves.setdefault(key, {"weight": 0.0, "vol_share": 0.0, "cvar_share": 0.0})
        s["weight"] += float(w[i])
        s["vol_share"] += float(rc[i] / sigma) if sigma > 0 else 0.0
        s["cvar_share"] += (
            float(cvar_contrib[i] / cvar_contrib.sum()) if cvar_contrib.sum() > 0 else 0.0
        )
    rows: list[dict[str, Any]] = [
        {
            "sleeve": k,
            **v,
            "budget": params.budgets.get(k),
            "usage": None if not params.budgets.get(k) else v["vol_share"] / params.budgets[k],
        }
        for k, v in sorted(sleeves.items(), key=lambda kv: -kv[1]["vol_share"])
    ]
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "ex_ante_vol": sigma,
            "target_vol": params.target_vol,
            "vol_target_exposure": params.target_vol / sigma if sigma > 0 else None,
            "rows": rows,
            "over_budget": [
                r["sleeve"] for r in rows if r["usage"] is not None and r["usage"] > 1.1
            ],
        },
    )


# ── IN-022 hedging designer ──────────────────────────────────────────────────


class HedgeParams(BaseModel):
    other: str | None = Field(None, max_length=20, description="Hedge index (default: benchmark)")
    target_beta: float = Field(0.5, ge=-1, le=2)
    put_moneyness: float = Field(0.95, gt=0.5, le=1.0)
    spread_moneyness: float = Field(0.85, gt=0.3, lt=1.0)
    days: int = Field(91, ge=14, le=730)


@analysis(
    id="IN-022",
    version="1.0.0",
    title="Hedging designer",
    description=(
        "Beta hedge: index short notional to reach the target beta (OLS β on 1 year of daily "
        "returns vs the benchmark), residual volatility; currency hedge: minimum-variance hedge "
        "ratio of the portfolio on each FX return; tail hedge: put spread (K1 = m1·S, K2 = m2·S) "
        "on the benchmark priced by BSM at the current implied vol (VIX-family close, else 1-year "
        "realised, stated), annual cost drag and payoff at −10/−20/−30 %; replay of stress windows "
        "with the hedge priced at each window start under the same assumption."
    ),
    applies_to={"portfolio"},
    params=HedgeParams,
    result_kind="table",
    requires={"portfolio", "other_prices", "vix"},
)
def hedging(inputs: AnalysisInputs, params: HedgeParams) -> AnalysisOutput:
    pf = _pf(inputs)
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError("Ingest the benchmark to design hedges (param `other`)")
    nav = pf.path.nav
    bench = other["adj_close"].astype(float)
    both = (
        pd.concat([nav.pct_change(), bench.pct_change()], axis=1, join="inner").dropna().iloc[-252:]
    )
    both.columns = ["p", "b"]
    beta = float(np.cov(both["p"], both["b"])[0, 1] / both["b"].var())
    value = float(pf.path.value.iloc[-1])
    hedge_notional = (beta - params.target_beta) * value
    resid = both["p"] - beta * both["b"]
    hedged = both["p"] - (beta - params.target_beta) * both["b"]
    fx_rows = []
    for ccy, series in (pf.fx or {}).items():
        fx_r = series.reindex(nav.index).ffill().pct_change()
        pair = pd.concat([nav.pct_change(), fx_r], axis=1, join="inner").dropna().iloc[-252:]
        if len(pair) > 60 and pair.iloc[:, 1].var() > 0:
            h = float(pair.cov().iloc[0, 1] / pair.iloc[:, 1].var())
            fx_rows.append(
                {
                    "currency": ccy,
                    "hedge_ratio": h,
                    "variance_reduction": float(
                        1 - (pair.iloc[:, 0] - h * pair.iloc[:, 1]).var() / pair.iloc[:, 0].var()
                    ),
                }
            )
    vix = inputs.data.get("vix")
    rf = float(inputs.conventions.get("rf") or 0.0)
    if isinstance(vix, pd.Series) and len(vix):
        vol_now, vol_source = (
            float(vix.iloc[-1]) / 100,
            f"{inputs.conventions.get('vix_symbol', 'VIX')} close",
        )
    else:
        vol_now, vol_source = (
            float(bench.pct_change().iloc[-252:].std() * math.sqrt(252)),
            "1-year realised (no VIX stored)",
        )
    s0, t = float(bench.iloc[-1]), params.days / 365
    k1, k2 = params.put_moneyness * s0, params.spread_moneyness * s0
    units = (beta * value) / s0  # protect the equity-like exposure
    cost = units * float(
        bsm_price("put", s0, k1, t, rf, 0.0, vol_now)
        - bsm_price("put", s0, k2, t, rf, 0.0, vol_now)
    )
    payoff = {
        f"{int(d * 100)}%": units * (max(k1 - s0 * (1 + d), 0) - max(k2 - s0 * (1 + d), 0))
        for d in (-0.1, -0.2, -0.3)
    }
    replay = []
    for name, start, end in STRESS_WINDOWS:
        window = bench.loc[
            (bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))
        ]
        pnav = nav.loc[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
        if len(window) < 2 or len(pnav) < 2:
            continue
        sb, eb = float(window.iloc[0]), float(window.iloc[-1])
        v0 = (
            float(vix.asof(pd.Timestamp(start))) / 100
            if isinstance(vix, pd.Series) and len(vix.loc[: pd.Timestamp(start)])
            else vol_now
        )
        u = beta / sb  # per 1 of portfolio value
        premium = u * float(
            bsm_price("put", sb, params.put_moneyness * sb, t, rf, 0.0, v0)
            - bsm_price("put", sb, params.spread_moneyness * sb, t, rf, 0.0, v0)
        )
        pay = u * (
            max(params.put_moneyness * sb - eb, 0) - max(params.spread_moneyness * sb - eb, 0)
        )
        port = float(pnav.iloc[-1] / pnav.iloc[0] - 1)
        replay.append(
            {
                "window": name,
                "portfolio": port,
                "hedged": port + pay - premium,
                "benchmark": eb / sb - 1,
            }
        )
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "benchmark": inputs.conventions.get("other_symbol"),
            "value": value,
            "beta": beta,
            "target_beta": params.target_beta,
            "hedge_notional": hedge_notional,
            "vol_unhedged": float(both["p"].std() * math.sqrt(252)),
            "vol_hedged": float(hedged.std() * math.sqrt(252)),
            "residual_vol": float(resid.std() * math.sqrt(252)),
            "currency": fx_rows,
            "tail": {
                "strikes": [k1, k2],
                "days": params.days,
                "implied_vol": vol_now,
                "vol_source": vol_source,
                "cost": cost,
                "cost_pct_of_value": cost / value,
                "annual_drag": cost / value * 365 / params.days,
                "payoff": payoff,
            },
            "replay": replay,
            "replay_assumption": (
                "Puts priced by BSM at each window start with the VIX-family close then "
                "(else today's vol)."
            ),
        },
    )


# ── IN-024 goal projection / IN-025 Kelly ────────────────────────────────────


class GoalParams(CmaParams):
    initial: float = Field(50_000, ge=0)
    monthly_contribution: float = Field(500, ge=0)
    years: int = Field(20, ge=1, le=60)
    goal: float = Field(300_000, gt=0)
    withdrawal_years: int = Field(25, ge=0, le=60)
    simulations: int = Field(2000, ge=200, le=20_000)


def _paths(mu: float, vol: float, months: int, sims: int, seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = math.log1p(mu) / 12 - vol**2 / 24
    return np.exp(rng.normal(drift, vol / math.sqrt(12), (sims, months)))


@analysis(
    id="IN-024",
    version="1.0.0",
    title="Goal projection",
    description=(
        "Monte Carlo (monthly lognormal, the profile's strategic allocation on the CMA) of the "
        "initial amount plus monthly contributions: probability of reaching the goal, terminal "
        "quantiles; sequence-of-returns risk (each path's months reordered worst-first vs "
        "worst-last: in accumulation, late losses hurt most); safe "
        "withdrawal rate = constant annual withdrawal (as % of the goal) that survives the "
        "withdrawal years in 90 % / 95 % of paths."
    ),
    applies_to={"profile"},
    params=GoalParams,
    result_kind="table",
    requires={"profile", "cma"},
    budget_ms=20_000,
)
def goal_projection(inputs: AnalysisInputs, params: GoalParams) -> AnalysisOutput:
    profile = _profile(inputs)
    score = glide_score(score_profile(profile["answers"])["risk_score"], None)
    weights = MODEL_ALLOCATIONS[score]
    cma_out = capital_market_assumptions(inputs, params)
    exp = {r["class"]: r["expected_return"] or 0.0 for r in cma_out["rows"]}
    mu = sum(w * exp.get(k, 0.0) for k, w in weights.items())
    cov = cma_out["cov"]
    if cov is None:
        raise InsufficientDataError("Ingest the asset-class data (POST /v1/invest/ingest)")
    keys = [k for k in weights if k in cov.index]
    wv = np.array([weights[k] for k in keys])
    vol = float(math.sqrt(wv @ cov.loc[keys, keys].to_numpy() @ wv))
    months = params.years * 12
    growth = _paths(mu, vol, months, params.simulations)
    wealth = np.full(params.simulations, params.initial)
    for m in range(months):
        wealth = (wealth + params.monthly_contribution) * growth[:, m]
    success = float((wealth >= params.goal).mean())
    order = np.sort(growth, axis=1)

    def run(g: np.ndarray) -> np.ndarray:
        w = np.full(g.shape[0], params.initial)
        for m in range(g.shape[1]):
            w = (w + params.monthly_contribution) * g[:, m]
        return w

    worst_first, worst_last = run(order), run(order[:, ::-1])
    swr: dict[str, float | None] = {}
    if params.withdrawal_years:
        dec = _paths(mu, vol, params.withdrawal_years * 12, params.simulations, seed=29)
        for level in (0.90, 0.95):
            lo, hi = 0.0, 0.2
            for _ in range(40):
                rate = (lo + hi) / 2
                w = np.full(params.simulations, params.goal)
                for m in range(dec.shape[1]):
                    w = np.maximum(w - params.goal * rate / 12, 0) * dec[:, m]
                if (w > 0).mean() >= level:
                    lo = rate
                else:
                    hi = rate
            swr[f"swr_{int(level * 100)}"] = lo
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "risk_score": score,
            "expected_return": mu,
            "volatility": vol,
            "probability_of_goal": success,
            "terminal_p5": float(np.quantile(wealth, 0.05)),
            "terminal_p50": float(np.quantile(wealth, 0.5)),
            "terminal_p95": float(np.quantile(wealth, 0.95)),
            "contributed": params.initial + params.monthly_contribution * months,
            "sequence_risk": {
                "worst_months_first_median": float(np.median(worst_first)),
                "worst_months_last_median": float(np.median(worst_last)),
            },
            **swr,
        },
    )


class KellyParams(BaseModel):
    fraction: float = Field(0.5, gt=0, le=1, description="Fractional Kelly (½ by default)")
    rf: float | None = Field(None, ge=-0.05, le=0.2)


@analysis(
    id="IN-025",
    version="1.0.0",
    title="Kelly sizing",
    description=(
        "Continuous-time Kelly fraction f* = (μ − r)/σ² from the subject's annualised log-return "
        "moments, its 95 % range from the standard error of μ (σ/√years), and the fractional Kelly "
        "allocation; warns that the edge is estimated and full Kelly is rarely appropriate."
    ),
    applies_to=PRICE_SUBJECTS,
    params=KellyParams,
    result_kind="table",
    requires={"prices"},
)
def kelly(inputs: AnalysisInputs, params: KellyParams) -> AnalysisOutput:
    close = _prices(inputs)[_field(inputs)].astype(float)
    r = np.log(close).diff().dropna()
    if len(r) < 252:
        raise InsufficientDataError(f"Kelly needs ≥ 252 sessions, got {len(r)}")
    a = int(inputs.conventions.get("annualisation") or 252)
    mu = float(r.mean() * a + r.var() * a / 2)  # arithmetic annual mean
    var = float(r.var() * a)
    rf = float(params.rf if params.rf is not None else inputs.conventions.get("rf") or 0.0)
    years = len(r) / a
    se = math.sqrt(var) / math.sqrt(years)
    full = (mu - rf) / var
    z = stats.norm.ppf(0.975)
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "mu": mu,
            "volatility": math.sqrt(var),
            "rf": rf,
            "years": years,
            "kelly_full": full,
            "kelly_low": (mu - z * se - rf) / var,
            "kelly_high": (mu + z * se - rf) / var,
            "kelly_fractional": params.fraction * full,
            "fraction": params.fraction,
            "warning": (
                "μ is estimated with a wide error band: the Kelly range spans it; "
                "size well below full Kelly."
            ),
        },
    )
