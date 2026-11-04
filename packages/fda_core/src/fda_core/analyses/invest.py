"""Investor profile, capital-market assumptions & strategic allocation (ANALYTICS_CATALOG IN-001…
IN-003; module M22). IN-004 (IPS) is a report built from these payloads.

Every payload carries `label = MODEL_LABEL`: these are model outputs for the owner's own use,
not personalised advice for third parties (PRD V4 rule). Scoring is a versioned table
(`SCORING_VERSION`): the same answers always give the same score, with each contribution shown.

Subjects: `profile` (id = stored profile) for IN-001/IN-003; IN-002 also runs on `market`.
Inputs: `profile` = {"answers": {...}, "name"}; `cma_prices` = {class: monthly-capable level
series of the long-history USD proxy}; `cma_series` = {code: level series} (DTB3, DGS10, IG/HY
OAS, T10YIE); `current_weights` = {class: weight} of a portfolio to compare (optional).
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import optimize

from fda_core.portfolio import covariance
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

MODEL_LABEL = "Model output — not personalised advice for third parties"
SCORING_VERSION = "profile-v1"
PROFILE: set[Any] = {"profile"}

# ── IN-001 questionnaire: question → (dimension, {answer: points}) ───────────
# Dimension C = capacity to bear risk, W = willingness. Points on a 1–7 scale.
QUESTIONS: dict[str, tuple[str, str, dict[str, int]]] = {
    "horizon": (
        "C",
        "Investment horizon",
        {"<2y": 1, "2-5y": 2, "5-10y": 4, "10-20y": 6, ">20y": 7},
    ),
    "income_stability": (
        "C",
        "Income stability",
        {"unstable": 1, "variable": 3, "stable": 5, "very_stable": 6},
    ),
    "emergency_fund": ("C", "Emergency fund", {"none": 1, "<3m": 3, "3-6m": 5, ">6m": 6}),
    "savings_rate": ("C", "Savings rate", {"<5%": 2, "5-15%": 4, ">15%": 6}),
    "debt": ("C", "Debt burden", {"high": 1, "moderate": 3, "low": 6}),
    "liquidity_needs": ("C", "Liquidity needs within 2 years", {">25%": 1, "some": 3, "none": 6}),
    "drawdown_reaction": (
        "W",
        "Your 100k falls to 70k in 6 months: you…",
        {"sell_all": 1, "sell_some": 2, "hold": 5, "buy_more": 7},
    ),
    "max_loss": (
        "W",
        "Largest yearly loss you would accept",
        {"5%": 1, "10%": 2, "20%": 4, "30%": 5, "40%+": 7},
    ),
    "preference": ("W", "Preference", {"stability": 1, "balanced": 4, "growth": 7}),
    "knowledge": ("W", "Investment knowledge", {"none": 2, "basic": 3, "good": 5, "expert": 6}),
    "experience": ("W", "Years of investing", {"0": 2, "1-3": 3, "3-10": 5, ">10": 6}),
}
MAX_DRAWDOWN = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.28, 6: 0.35, 7: 0.45}
STATED_LOSS = {"5%": 0.05, "10%": 0.10, "20%": 0.20, "30%": 0.30, "40%+": 0.45}


def score_profile(answers: dict[str, Any]) -> dict[str, Any]:
    """Capacity and willingness = rounded mean points (1–7); risk score = the lower of the two
    (the more prudent wins, explained); drawdown tolerance = min(score table, stated loss)."""
    missing = [q for q in QUESTIONS if q not in answers]
    if missing:
        raise InsufficientDataError(f"Unanswered questions: {', '.join(missing)}")
    contributions: list[dict[str, Any]] = []
    points: dict[str, list[int]] = {"C": [], "W": []}
    for key, (dim, text, options) in QUESTIONS.items():
        answer = str(answers[key])
        if answer not in options:
            raise InsufficientDataError(
                f"{key}: unknown answer {answer!r} (options: {', '.join(options)})"
            )
        points[dim].append(options[answer])
        contributions.append(
            {
                "question": key,
                "text": text,
                "dimension": "capacity" if dim == "C" else "willingness",
                "answer": answer,
                "points": options[answer],
            }
        )
    capacity = int(np.clip(round(float(np.mean(points["C"]))), 1, 7))
    willingness = int(np.clip(round(float(np.mean(points["W"]))), 1, 7))
    score = min(capacity, willingness)
    binding = (
        "capacity"
        if capacity < willingness
        else "willingness"
        if willingness < capacity
        else "both"
    )
    return {
        "scoring_version": SCORING_VERSION,
        "capacity": capacity,
        "willingness": willingness,
        "risk_score": score,
        "binding": binding,
        "explanation": (
            f"Capacity {capacity} and willingness {willingness}: the lower one sets the score."
            if binding != "both"
            else f"Capacity and willingness agree at {score}."
        ),
        "max_drawdown_tolerance": min(MAX_DRAWDOWN[score], STATED_LOSS[str(answers["max_loss"])]),
        "contributions": contributions,
    }


def required_return(
    goal: float, years: float, current: float, monthly_saving: float
) -> float | None:
    """Annual return r such that current·(1+r)^n + saving·Σ(1+m)^k reaches the goal (monthly
    compounding m = (1+r)^(1/12) − 1); None when unreachable below 100 %/yr."""
    if goal <= 0 or years <= 0:
        return None
    months = round(years * 12)

    def future(r: float) -> float:
        m = math.pow(1 + r, 1 / 12) - 1
        annuity = float(months) if abs(m) < 1e-12 else (math.pow(1 + m, months) - 1) / m
        return current * math.pow(1 + r, years) + monthly_saving * annuity - goal

    if future(-0.5) > 0:
        return -0.5
    if future(1.0) < 0:
        return None
    root: float = optimize.brentq(future, -0.5, 1.0, xtol=1e-10)
    return float(root)


class ProfileParams(BaseModel):
    pass


def _profile(inputs: AnalysisInputs) -> dict[str, Any]:
    profile = inputs.data.get("profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("answers"), dict):
        raise InsufficientDataError("No investor profile: fill in the questionnaire")
    return profile


@analysis(
    id="IN-001",
    version="1.0.0",
    title="Investor profile",
    description=(
        "Questionnaire (11 questions, scoring table profile-v1): capacity and willingness on "
        "1–7 (rounded mean of points), risk score = the lower one; drawdown tolerance = min(score "
        "table, stated loss); required return to reach the goal from current assets and monthly "
        "savings (monthly compounding). Model output, not advice."
    ),
    applies_to=PROFILE,
    params=ProfileParams,
    result_kind="table",
    requires={"profile"},
)
def investor_profile(inputs: AnalysisInputs, params: ProfileParams) -> AnalysisOutput:
    profile = _profile(inputs)
    answers = profile["answers"]
    scored = score_profile(answers)
    goal = answers.get("goal") or {}
    required = None
    if goal:
        required = required_return(
            float(goal.get("amount", 0)),
            float(goal.get("years", 0)),
            float(goal.get("current", 0)),
            float(goal.get("monthly_saving", 0)),
        )
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "name": profile.get("name"),
            **scored,
            "goal": goal or None,
            "required_return": required,
            "wrapper": answers.get("wrapper"),
            "rows": scored["contributions"],
        },
    )


# ── IN-002 capital-market assumptions ────────────────────────────────────────

ClassKey = Literal[
    "cash",
    "govt_bonds",
    "ig_credit",
    "hy_credit",
    "us_equity",
    "devx_equity",
    "em_equity",
    "reits",
    "gold",
    "commodities",
    "crypto",
]
# class → (label, long-history USD proxy, kind)
CLASSES: dict[str, tuple[str, str | None, str]] = {
    "cash": ("Cash", None, "cash"),
    "govt_bonds": ("Government bonds (7–10y)", "IEF", "bond"),
    "ig_credit": ("Investment-grade credit", "LQD", "bond"),
    "hy_credit": ("High-yield credit", "HYG", "bond"),
    "us_equity": ("US equities", "SPY", "equity"),
    "devx_equity": ("Developed ex-US equities", "EFA", "equity"),
    "em_equity": ("Emerging-market equities", "EEM", "equity"),
    "reits": ("Real estate (REITs)", "VNQ", "equity"),
    "gold": ("Gold", "GLD", "real"),
    "commodities": ("Commodities", "DBC", "real"),
    "crypto": ("Crypto (BTC)", "BTC-USD", "crypto"),
}
CMA_SERIES = ("DTB3", "DGS10", "BAMLC0A0CM", "BAMLH0A0HYM2", "T10YIE")
# Implementation examples per class (labels only — check each fund's KID/prospectus).
IMPLEMENTATION: dict[str, dict[str, str]] = {
    "cash": {
        "USD": "BIL — SPDR 1-3 Month T-Bill",
        "EUR": "XEON.DE — Xtrackers EUR Overnight Rate Swap (acc)",
    },
    "govt_bonds": {
        "USD": "IEF — iShares 7-10 Year Treasury",
        "EUR": "IEGA.AS — iShares Core € Govt Bond (dist)",
    },
    "ig_credit": {
        "USD": "LQD — iShares iBoxx $ IG Corporate",
        "EUR": "IEAC.AS — iShares Core € Corp Bond (dist)",
    },
    "hy_credit": {
        "USD": "HYG — iShares iBoxx $ High Yield",
        "EUR": "IHYG.L — iShares € High Yield Corp Bond",
    },
    "us_equity": {
        "USD": "VTI — Vanguard Total Stock Market",
        "EUR": "SXR8.DE — iShares Core S&P 500 (acc)",
    },
    "devx_equity": {
        "USD": "IEFA — iShares Core MSCI EAFE",
        "EUR": "IWDA.AS — iShares Core MSCI World (acc; includes US)",
    },
    "em_equity": {
        "USD": "IEMG — iShares Core MSCI EM",
        "EUR": "EMIM.AS — iShares Core MSCI EM IMI (acc)",
    },
    "reits": {
        "USD": "VNQ — Vanguard Real Estate",
        "EUR": "IWDP.AS — iShares Developed Markets Property Yield",
    },
    "gold": {"USD": "GLD — SPDR Gold Shares", "EUR": "SGLN.L — iShares Physical Gold ETC"},
    "commodities": {
        "USD": "DBC — Invesco DB Commodity Index",
        "EUR": "CMOD.L — Invesco Bloomberg Commodity",
    },
    "crypto": {
        "USD": "IBIT — iShares Bitcoin Trust",
        "EUR": "BTCE.DE — ETC Group Physical Bitcoin",
    },
}


class CmaParams(BaseModel):
    method: Literal["blend", "historical", "building_block"] = "blend"
    real_growth: float = Field(0.015, ge=-0.02, le=0.05, description="Real EPS growth for equities")
    valuation_drift: float = Field(0.0, ge=-0.05, le=0.05, description="Annual valuation change")
    hy_default_loss: float = Field(0.02, ge=0, le=0.1)
    overrides: dict[str, float] = Field(default_factory=dict, description="class → expected return")
    shrinkage: Literal["ledoit_wolf", "oas", "sample"] = "ledoit_wolf"


def _last(series: dict[str, pd.Series], code: str) -> float | None:
    s = series.get(code)
    return None if s is None or s.empty else float(s.iloc[-1]) / 100  # percent → decimal


def capital_market_assumptions(inputs: AnalysisInputs, params: CmaParams) -> dict[str, Any]:
    """Monthly returns of each proxy over its full stored history; historical = geometric
    annual mean with its standard error; building-block = yield + growth (+ drift) per kind;
    blend = mean of the two when both exist; covariance on the common window, shrunk."""
    raw_prices, raw_series = inputs.data.get("cma_prices"), inputs.data.get("cma_series")
    prices: dict[str, pd.Series] = raw_prices if isinstance(raw_prices, dict) else {}
    series: dict[str, pd.Series] = raw_series if isinstance(raw_series, dict) else {}
    rf = _last(series, "DTB3")
    y10 = _last(series, "DGS10")
    ig, hy = _last(series, "BAMLC0A0CM"), _last(series, "BAMLH0A0HYM2")
    infl = _last(series, "T10YIE")
    monthly: dict[str, pd.Series] = {}
    rows = []
    for key, (label, proxy, kind) in CLASSES.items():
        hist_mu: float | None = None
        hist_se: float | None = None
        expected: float | None
        years = 0.0
        if proxy and proxy in prices:
            p = prices[proxy].astype(float).dropna()
            p.index = pd.to_datetime(p.index)
            m = p.resample("ME").last().pct_change().dropna()
            if len(m) >= 36:
                monthly[key] = m
                years = len(m) / 12
                log_mean = float(np.log1p(m).mean())
                hist_mu = math.expm1(12 * log_mean)
                hist_se = float(np.log1p(m).std(ddof=1) * math.sqrt(12) / math.sqrt(years))
        bb = None
        dividend = (inputs.data.get("cma_yields") or {}).get(proxy) if proxy else None
        if kind == "cash":
            bb = rf
        elif key == "govt_bonds":
            bb = y10
        elif key == "ig_credit" and y10 is not None and ig is not None:
            bb = y10 + ig - 0.002  # spread net of ~20 bp expected credit loss
        elif key == "hy_credit" and y10 is not None and hy is not None:
            bb = y10 + hy - params.hy_default_loss
        elif kind == "equity" and infl is not None:
            bb = (dividend or 0.02) + params.real_growth + infl + params.valuation_drift
        elif kind == "real" and infl is not None:
            bb = infl
        if key in params.overrides:
            expected, source = float(params.overrides[key]), "override"
        elif params.method == "historical" or (params.method == "blend" and bb is None):
            expected, source = hist_mu, "historical"
        elif params.method == "building_block" or hist_mu is None:
            expected, source = bb, "building-block"
        else:
            expected, source = (hist_mu + (bb or 0.0)) / 2, "blend"
        if key == "cash" and expected is None:
            expected, source = 0.0, "no short rate stored"
        vol = (
            float(monthly[key].std(ddof=1) * math.sqrt(12))
            if key in monthly
            else (0.005 if key == "cash" else None)
        )
        rows.append(
            {
                "class": key,
                "label": label,
                "proxy": proxy,
                "expected_return": expected,
                "source": source,
                "historical": hist_mu,
                "historical_se": hist_se,
                "building_block": bb,
                "volatility": vol,
                "years": round(years, 1),
                "range_low": None
                if expected is None or hist_se is None
                else expected - 1.96 * hist_se,
                "range_high": None
                if expected is None or hist_se is None
                else expected + 1.96 * hist_se,
            }
        )
    keys = [k for k in CLASSES if k in monthly]
    corr = cov = None
    window = None
    if len(keys) >= 2:
        frame = pd.concat({k: monthly[k] for k in keys}, axis=1).dropna()
        if len(frame) >= 36:
            window = (
                frame.index[0].date().isoformat(),
                frame.index[-1].date().isoformat(),
                len(frame),
            )
            sigma = covariance(frame, params.shrinkage) * 12
            d = np.sqrt(np.diag(sigma))
            cov = pd.DataFrame(sigma, index=keys, columns=keys)
            corr = pd.DataFrame(sigma / np.outer(d, d), index=keys, columns=keys)
    return {
        "rows": rows,
        "cov": cov,
        "corr": corr,
        "window": window,
        "rf": rf,
        "as_of": max((s.index[-1] for s in prices.values() if len(s)), default=None),
    }


def _cma_payload(cma: dict[str, Any]) -> dict[str, Any]:
    corr = cma["corr"]
    return {
        "label": MODEL_LABEL,
        "caveat": "Expected returns carry large estimation error: read the ranges.",
        "rows": cma["rows"],
        "risk_free": cma["rf"],
        "covariance_window": cma["window"],
        "as_of": None if cma["as_of"] is None else pd.Timestamp(cma["as_of"]).date().isoformat(),
        **(
            {
                "grid": {
                    "rows": list(corr.index),
                    "columns": list(corr.columns),
                    "values": corr.round(4).to_numpy().tolist(),
                    "unit": "number",
                    "best": [],
                }
            }
            if corr is not None
            else {}
        ),
    }


@analysis(
    id="IN-002",
    version="1.0.0",
    title="Capital-market assumptions",
    description=(
        "Per asset class (long-history USD ETF proxies): historical geometric mean of monthly "
        "returns with its ±1.96 SE range; building-block = cash: T-bill; govt: 10y yield; IG: "
        "10y + OAS − 20 bp; HY: 10y + OAS − default loss; equities: dividend yield + real growth + "
        "breakeven inflation (+ valuation drift); real assets: breakeven; blend = average; user "
        "overrides. Volatility and correlation from monthly returns on the common window with "
        "Ledoit–Wolf shrinkage (annualised)."
    ),
    applies_to={"profile", "market"},
    params=CmaParams,
    result_kind="table",
    requires={"cma"},
)
def cma(inputs: AnalysisInputs, params: CmaParams) -> AnalysisOutput:
    out = capital_market_assumptions(inputs, params)
    if not any(r["expected_return"] is not None for r in out["rows"] if r["class"] != "cash"):
        raise InsufficientDataError(
            "Ingest the asset-class proxies and rates (POST /v1/invest/ingest)"
        )
    return AnalysisOutput("table", _cma_payload(out))


# ── IN-003 strategic allocation ──────────────────────────────────────────────

MODEL_ALLOCATIONS: dict[int, dict[str, float]] = {
    1: {
        "cash": 0.45,
        "govt_bonds": 0.30,
        "ig_credit": 0.15,
        "us_equity": 0.05,
        "devx_equity": 0.03,
        "gold": 0.02,
    },
    2: {
        "cash": 0.15,
        "govt_bonds": 0.35,
        "ig_credit": 0.20,
        "hy_credit": 0.05,
        "us_equity": 0.12,
        "devx_equity": 0.08,
        "em_equity": 0.02,
        "gold": 0.03,
    },
    3: {
        "cash": 0.10,
        "govt_bonds": 0.25,
        "ig_credit": 0.15,
        "hy_credit": 0.05,
        "us_equity": 0.22,
        "devx_equity": 0.12,
        "em_equity": 0.04,
        "reits": 0.03,
        "gold": 0.04,
    },
    4: {
        "cash": 0.05,
        "govt_bonds": 0.17,
        "ig_credit": 0.12,
        "hy_credit": 0.06,
        "us_equity": 0.30,
        "devx_equity": 0.15,
        "em_equity": 0.06,
        "reits": 0.04,
        "gold": 0.05,
    },
    5: {
        "cash": 0.03,
        "govt_bonds": 0.10,
        "ig_credit": 0.08,
        "hy_credit": 0.06,
        "us_equity": 0.37,
        "devx_equity": 0.18,
        "em_equity": 0.08,
        "reits": 0.04,
        "gold": 0.05,
        "crypto": 0.01,
    },
    6: {
        "govt_bonds": 0.06,
        "ig_credit": 0.05,
        "hy_credit": 0.05,
        "us_equity": 0.43,
        "devx_equity": 0.20,
        "em_equity": 0.10,
        "reits": 0.05,
        "gold": 0.04,
        "commodities": 0.01,
        "crypto": 0.01,
    },
    7: {
        "govt_bonds": 0.02,
        "hy_credit": 0.03,
        "us_equity": 0.48,
        "devx_equity": 0.22,
        "em_equity": 0.13,
        "reits": 0.05,
        "gold": 0.03,
        "commodities": 0.02,
        "crypto": 0.02,
    },
}
# Published expected-volatility band per score (checked against the CMA, acceptance 2).
VOL_BANDS: dict[int, tuple[float, float]] = {
    1: (0.01, 0.05),
    2: (0.03, 0.07),
    3: (0.05, 0.09),
    4: (0.07, 0.11),
    5: (0.09, 0.13),
    6: (0.11, 0.15),
    7: (0.13, 0.18),
}
CORE = {"cash", "govt_bonds", "ig_credit", "us_equity", "devx_equity"}


def glide_score(score: int, years_to_goal: float | None) -> int:
    """Within 10 years of a dated goal, one step down per 3 years closer (never below 1)."""
    if years_to_goal is None or years_to_goal >= 10:
        return score
    return max(1, score - math.ceil((10 - years_to_goal) / 3))


def allocation_stats(weights: dict[str, float], cma_out: dict[str, Any]) -> dict[str, Any]:
    exp = {r["class"]: r["expected_return"] for r in cma_out["rows"]}
    cov = cma_out["cov"]
    mu = sum(w * (exp.get(k) or 0.0) for k, w in weights.items())
    vol = None
    if cov is not None:
        keys = [k for k in weights if k in cov.index]
        w = np.array([weights[k] for k in keys])
        vol = float(math.sqrt(max(w @ cov.loc[keys, keys].to_numpy() @ w, 0.0)))
        uncovered = sum(v for k, v in weights.items() if k not in cov.index and k != "cash")
        if uncovered > 0.02:
            vol = None  # a material class has no covariance: no honest number
    return {"expected_return": mu, "volatility": vol}


def monte_carlo_drawdown(
    weights: dict[str, float],
    cma_out: dict[str, Any],
    years: float = 10,
    paths: int = 2000,
    seed: int = 7,
) -> dict[str, float] | None:
    """Monthly lognormal paths with the CMA means and covariance: distribution of the maximum
    drawdown over the horizon (5/50/95 %) and of the terminal value per 1 invested."""
    cov = cma_out["cov"]
    if cov is None:
        return None
    exp = {r["class"]: r["expected_return"] or 0.0 for r in cma_out["rows"]}
    keys = [k for k in weights if k in cov.index]
    if not keys:
        return None
    w = np.array([weights[k] for k in keys])
    w = w / w.sum()
    mu_p = float(sum(w[i] * exp[k] for i, k in enumerate(keys)))
    var_p = float(w @ cov.loc[keys, keys].to_numpy() @ w)
    months = int(years * 12)
    rng = np.random.default_rng(seed)
    drift = math.log1p(mu_p) / 12 - var_p / 24
    shocks = rng.normal(drift, math.sqrt(var_p / 12), (paths, months))
    wealth = np.exp(np.cumsum(shocks, axis=1))
    peak = np.maximum.accumulate(np.concatenate([np.ones((paths, 1)), wealth], axis=1), axis=1)[
        :, 1:
    ]
    mdd = (wealth / peak - 1).min(axis=1)
    return {
        "mdd_p5": float(np.quantile(mdd, 0.05)),
        "mdd_p50": float(np.quantile(mdd, 0.5)),
        "mdd_p95": float(np.quantile(mdd, 0.95)),
        "terminal_p5": float(np.quantile(wealth[:, -1], 0.05)),
        "terminal_p50": float(np.quantile(wealth[:, -1], 0.5)),
        "terminal_p95": float(np.quantile(wealth[:, -1], 0.95)),
        "years": years,
        "paths": paths,
    }


def allocation_backtest(weights: dict[str, float], inputs: AnalysisInputs) -> dict[str, Any] | None:
    """Model weights rebalanced monthly on the proxies' monthly returns (cash at the T-bill
    rate) vs an equal-weight mix of the same classes, over the common history: CAGR, volatility,
    max drawdown, Sharpe. Fixed weights, so no estimation look-ahead; history is in-sample for
    the CMA but not for the weights, which are a published table."""
    raw = inputs.data.get("cma_prices")
    prices: dict[str, pd.Series] = raw if isinstance(raw, dict) else {}
    series = (
        inputs.data.get("cma_series") if isinstance(inputs.data.get("cma_series"), dict) else {}
    )
    monthly: dict[str, pd.Series] = {}
    for key in weights:
        proxy = CLASSES[key][1]
        if proxy and proxy in prices:
            s = prices[proxy].astype(float).dropna()
            s.index = pd.to_datetime(s.index)
            monthly[key] = s.resample("ME").last().pct_change()
        elif key == "cash" and isinstance(series, dict) and "DTB3" in series:
            rate = series["DTB3"].astype(float)
            rate.index = pd.to_datetime(rate.index)
            monthly[key] = (rate.resample("ME").last() / 100 / 12).shift(1)
    if set(monthly) != set(weights):
        return None
    frame = pd.DataFrame(monthly).dropna()
    if len(frame) < 36:
        return None
    w = np.array([weights[k] for k in frame.columns])
    eq = np.full(len(w), 1 / len(w))
    rf = float(frame["cash"].mean() * 12) if "cash" in frame else 0.0

    def stats_of(r: pd.Series) -> dict[str, float]:
        nav = (1 + r).cumprod()
        years = len(r) / 12
        vol = float(r.std(ddof=1) * math.sqrt(12))
        cagr = float(nav.iloc[-1] ** (1 / years) - 1)
        return {
            "cagr": cagr,
            "volatility": vol,
            "sharpe": (cagr - rf) / vol if vol > 0 else math.nan,
            "max_drawdown": float((nav / nav.cummax() - 1).min()),
        }

    model = stats_of(pd.Series(frame.to_numpy() @ w, index=frame.index))
    equal = stats_of(pd.Series(frame.to_numpy() @ eq, index=frame.index))
    return {
        "from": frame.index[0].date().isoformat(),
        "to": frame.index[-1].date().isoformat(),
        "months": len(frame),
        "model": model,
        "equal_weight": equal,
        "beats_equal_weight_sharpe": model["sharpe"] > equal["sharpe"],
    }


class AllocationParams(CmaParams):
    risk_score: int | None = Field(None, ge=1, le=7, description="Override the profile's score")
    currency: Literal["EUR", "USD"] = "EUR"
    compare: int | None = Field(None, description="Portfolio id to compare with")


@analysis(
    id="IN-003",
    version="1.1.0",
    title="Strategic allocation",
    description=(
        "Model allocation for the risk score (glide path: one step down per 3 years inside 10 "
        "years of a dated goal), core/satellite split, ETF implementation examples (UCITS for "
        "EUR investors), expected return and volatility from the CMA (checked against the "
        "score's published band), Monte Carlo maximum-drawdown and terminal-value ranges, monthly "
        "backtest of the model weights vs an equal-weight mix of the same classes, gap "
        "to a current portfolio by asset class. Model output, not advice."
    ),
    applies_to=PROFILE,
    params=AllocationParams,
    result_kind="table",
    requires={"profile", "cma"},
)
def strategic_allocation(inputs: AnalysisInputs, params: AllocationParams) -> AnalysisOutput:
    profile = _profile(inputs)
    scored = score_profile(profile["answers"])
    goal = profile["answers"].get("goal") or {}
    years = float(goal["years"]) if goal.get("years") else None
    base = params.risk_score or scored["risk_score"]
    score = glide_score(base, years)
    weights = MODEL_ALLOCATIONS[score]
    cma_out = capital_market_assumptions(inputs, params)
    stats = allocation_stats(weights, cma_out)
    band = VOL_BANDS[score]
    vol = stats["volatility"]
    mc = monte_carlo_drawdown(weights, cma_out, min(years or 10, 30))
    current = (
        inputs.data.get("current_weights")
        if isinstance(inputs.data.get("current_weights"), dict)
        else None
    )
    rows = [
        {
            "class": k,
            "label": CLASSES[k][0],
            "weight": w,
            "sleeve": "core" if k in CORE else "satellite",
            "implementation": IMPLEMENTATION[k][params.currency],
            "current": None if current is None else current.get(k, 0.0),
            "gap": None if current is None else w - current.get(k, 0.0),
        }
        for k, w in sorted(weights.items(), key=lambda kv: -kv[1])
    ]
    if current and current.get("unclassified"):
        rows.append(
            {
                "class": "unclassified",
                "label": "Not mapped to a class",
                "weight": 0.0,
                "sleeve": "—",
                "implementation": "",
                "current": current["unclassified"],
                "gap": -current["unclassified"],
            }
        )
    path = (
        [
            {
                "years_to_goal": y,
                "risk_score": glide_score(base, y),
                "equity_share": sum(
                    v
                    for k, v in MODEL_ALLOCATIONS[glide_score(base, y)].items()
                    if CLASSES[k][2] == "equity"
                ),
            }
            for y in (10, 8, 6, 4, 2, 0)
        ]
        if years is not None
        else []
    )
    return AnalysisOutput(
        "table",
        {
            "label": MODEL_LABEL,
            "profile_score": base,
            "risk_score": score,
            "glide_applied": score != base,
            "max_drawdown_tolerance": scored["max_drawdown_tolerance"],
            "currency": params.currency,
            "core_share": sum(w for k, w in weights.items() if k in CORE),
            "expected_return": stats["expected_return"],
            "volatility": vol,
            "vol_band": list(band),
            "within_band": None if vol is None else band[0] <= vol <= band[1],
            "monte_carlo": mc,
            "historical_evidence": allocation_backtest(weights, inputs),
            "drawdown_check": None
            if mc is None
            else (
                "within tolerance"
                if -mc["mdd_p50"] <= scored["max_drawdown_tolerance"]
                else "median drawdown exceeds tolerance"
            ),
            "rows": rows,
            "glide_path": path,
            "compared_portfolio": params.compare,
        },
    )


# ── holdings → asset classes (gap analysis) ──────────────────────────────────

KNOWN_ETFS: dict[str, str] = {
    "BIL": "cash",
    "SHV": "cash",
    "XEON.DE": "cash",
    "IEF": "govt_bonds",
    "TLT": "govt_bonds",
    "GOVT": "govt_bonds",
    "IEGA.AS": "govt_bonds",
    "AGG": "ig_credit",
    "BND": "ig_credit",
    "LQD": "ig_credit",
    "IEAC.AS": "ig_credit",
    "IEAG.AS": "ig_credit",
    "HYG": "hy_credit",
    "JNK": "hy_credit",
    "IHYG.L": "hy_credit",
    "SPY": "us_equity",
    "IVV": "us_equity",
    "VOO": "us_equity",
    "VTI": "us_equity",
    "QQQ": "us_equity",
    "SXR8.DE": "us_equity",
    "CSPX.L": "us_equity",
    "EFA": "devx_equity",
    "IEFA": "devx_equity",
    "VGK": "devx_equity",
    "EXSA.DE": "devx_equity",
    "IWDA.AS": "devx_equity",
    "URTH": "devx_equity",
    "EEM": "em_equity",
    "IEMG": "em_equity",
    "EMIM.AS": "em_equity",
    "VWO": "em_equity",
    "VNQ": "reits",
    "IWDP.AS": "reits",
    "GLD": "gold",
    "IAU": "gold",
    "SGLN.L": "gold",
    "DBC": "commodities",
    "CMOD.L": "commodities",
    "IBIT": "crypto",
    "BTCE.DE": "crypto",
}


def classify_holding(symbol: str, asset_class: str | None, country: str | None) -> str:
    """Known ETFs by ticker; single stocks by listing country (US vs other developed);
    crypto by class; anything else 'unclassified' (shown, never guessed)."""
    upper = symbol.upper()
    if upper in KNOWN_ETFS:
        return KNOWN_ETFS[upper]
    if upper == "CASH" or asset_class == "cash":
        return "cash"
    if asset_class == "crypto":
        return "crypto"
    if asset_class == "equity":
        return (
            "us_equity"
            if (country or "").upper() in ("US", "USA", "UNITED STATES")
            else "devx_equity"
        )
    return "unclassified"
