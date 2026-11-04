"""Valuation (ANALYTICS_CATALOG V-001…V-020; module M11). Defaults follow DECISIONS
D-VA-1…D-VA-6: FCFF two-stage DCF (5 explicit years + 5 fade years + terminal), mid-year
discounting, Gordon and exit-multiple terminal values, ERP 5 %, assumptions seeded from history.

Inputs: `fundamentals` (PIT facts, latest filing), `fundamentals_all` (every filing, for
point-in-time history: a value becomes known on its first filing date), `prices`, optional
`other_prices` (benchmark, for β), conventions `rf` (latest DTB3 / €STR when stored).
These are model outputs from stated assumptions, not price targets.
"""

import math
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import optimize

from fda_core.analyses.ratios import Book, div, invested_capital, tax_rate
from fda_core.analyses.statements import COMPANY, Basis, _facts, build_periods
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

ERP_DEFAULT = 0.05
REQUIRES = {"fundamentals", "prices"}


def _f(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


# ── shared snapshot ──────────────────────────────────────────────────────────

# build_periods per stored fundamentals frame: the football field (V-016) and the methods it
# combines build the same Snapshot several times. Keyed by the frame's identity (the reference is
# kept, so ids cannot be reused while cached); bounded; callers get copies.
_PERIODS: dict[tuple[int, str], tuple[object, pd.DataFrame]] = {}


def _periods(inputs: AnalysisInputs, facts: pd.DataFrame, basis: Basis) -> pd.DataFrame:
    source = inputs.data.get("fundamentals")
    key = (id(source), basis)
    hit = _PERIODS.get(key)
    if hit is None or hit[0] is not source:
        if len(_PERIODS) >= 16:
            _PERIODS.clear()
        hit = (source, build_periods(facts, basis))
        _PERIODS[key] = hit
    return hit[1].copy()


# ── share count (feature 274) ────────────────────────────────────────────────

SHARE_MAX_AGE_DAYS = 400  # a count older than this vs the latest results is not current
SHARE_TOLERANCE = 0.20  # largest gap to the EPS-implied count for a share line to be trusted
SHARE_CANDIDATES = ("shares_outstanding", "shares_diluted_wavg")


class ShareCount(NamedTuple):
    value: float
    source: str
    as_of: str | None
    candidates: dict[str, float | None]
    eps_implied: float | None
    notes: tuple[str, ...]


def _eps_implied(*books: Book) -> float | None:
    """Net income to common ÷ diluted EPS on the latest period of the first book having both."""
    for book in books:
        if not book.periods:
            continue
        ni = book["net_income_common"].fillna(book["net_income"]).iloc[-1]
        eps = book["eps_diluted"].iloc[-1]
        if np.isfinite(ni) and np.isfinite(eps) and eps != 0 and ni / eps > 0:
            return float(ni / eps)
    return None


def resolve_shares(facts: pd.DataFrame, *books: Book) -> ShareCount:
    """Current share count for the market capitalisation (spec 274).

    Only counts dated within SHARE_MAX_AGE_DAYS of the latest net income count. Multi-class
    issuers (Visa, Berkshire, Alphabet) tag counts per class, so one line can miss classes or be
    in other units: the first of shares outstanding / diluted weighted average within ±20 % of
    the EPS-implied count wins, else the EPS-implied count itself is used.
    """
    frame = facts[facts["value"].notna()]
    ends = pd.to_datetime(frame["period_end"])
    income = ends[frame["line_item"] == "net_income"]
    reference = (income if len(income) else ends).max()
    current: dict[str, tuple[float, pd.Timestamp]] = {}
    stale: list[str] = []
    for code in SHARE_CANDIDATES:
        rows = frame[(frame["line_item"] == code) & (frame["value"] > 0)]
        if rows.empty:
            continue
        rows = rows.assign(_end=pd.to_datetime(rows["period_end"])).sort_values(
            ["_end", "filed_at"]
        )
        value, end = float(rows["value"].iloc[-1]), rows["_end"].iloc[-1]
        if (reference - end).days > SHARE_MAX_AGE_DAYS:
            stale.append(f"{code} ({end.date().isoformat()})")
        else:
            current[code] = (value, end)
    implied = _eps_implied(*books)
    candidates = {
        code: (current[code][0] if code in current else None) for code in SHARE_CANDIDATES
    }
    notes: list[str] = []
    divergent = [
        f"{code} {value / 1e6:,.0f} M"
        for code, (value, _) in current.items()
        if implied and abs(value / implied - 1) > SHARE_TOLERANCE
    ]
    if divergent:
        notes.append(
            f"Share lines diverge by more than 20 % from the EPS-implied count "
            f"({implied / 1e6 if implied else 0:,.0f} M): " + ", ".join(divergent) + "."
        )
    for code, (value, end) in current.items():
        if implied is None or abs(value / implied - 1) <= SHARE_TOLERANCE:
            if implied is None:
                notes.append(f"Share count ({code}) not cross-checked: no diluted EPS stored.")
            return ShareCount(
                value, code, end.date().isoformat(), candidates, implied, tuple(notes)
            )
    if implied is not None:
        notes.append("Market capitalisation uses the EPS-implied share count.")
        return ShareCount(implied, "eps_implied", None, candidates, implied, tuple(notes))
    detail = f" (latest: {', '.join(stale)})" if stale else ""
    raise InsufficientDataError(
        f"No current share count stored{detail}: market capitalisation unknown"
    )


class Snapshot:
    """Latest market and fundamental figures (TTM when four quarters exist, else last FY)."""

    def __init__(self, inputs: AnalysisInputs) -> None:
        facts = _facts(inputs)
        prices = inputs.data.get("prices")
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise InsufficientDataError("Valuation needs stored prices")
        self.conventions = inputs.conventions
        self.price = float(prices["close"].iloc[-1])
        self.price_date = pd.Timestamp(prices.index[-1]).date().isoformat()
        self.fy = Book(_periods(inputs, facts, "FY"))
        self.flow, self.basis = self.fy, "FY"
        try:
            ttm = _periods(inputs, facts, "TTM")
            complete = ttm.shape[1] and all(
                code in ttm.index and np.isfinite(ttm.loc[code].iloc[-1])
                for code in ("revenue", "net_income", "operating_income")
            )
            if complete:  # partial quarterly coverage (e.g. yfinance) → last fiscal year
                self.flow, self.basis = Book(ttm), "TTM"
        except InsufficientDataError:
            pass
        last = self.flow.periods[-1]
        self.period = last

        def flow(code: str) -> float:
            return float(self.flow[code].iloc[-1])

        def stock(code: str) -> float:
            series = self.flow[code].dropna()
            return float(series.iloc[-1]) if len(series) else float("nan")

        self.flow_value, self.stock_value = flow, stock
        self.share_count = resolve_shares(facts, self.flow, self.fy)
        self.shares = self.share_count.value
        self.market_cap = self.price * self.shares

        def z(code: str) -> float:
            value = stock(code)
            return value if np.isfinite(value) else 0.0

        self.debt = (
            z("short_term_debt")
            + z("current_portion_lt_debt")
            + z("long_term_debt")
            + z("operating_lease_liabilities")
        )
        self.cash = z("cash_and_equivalents") + z("short_term_investments")
        self.minorities = z("minority_interest_bs")
        self.preferred = z("preferred_equity")
        self.ev = self.market_cap + self.debt + self.preferred + self.minorities - self.cash
        self.net_debt = self.debt - self.cash
        rf = inputs.conventions.get("rf")
        self.rf = float(rf) if rf is not None else 0.04
        self.rf_source = (
            inputs.conventions.get("rf_source") or "default 4 % (no rate series stored)"
        )
        currency = inputs.conventions.get("currency") or "USD"
        self.currency = currency
        self.g_long = min(0.035, self.rf) if currency == "USD" else 0.03  # D-VA-5


def _beta(inputs: AnalysisInputs) -> tuple[float, str]:
    """5-year weekly regression β vs the benchmark, Blume-adjusted (bottom-up β with J8 peers)."""
    prices, bench = inputs.data.get("prices"), inputs.data.get("other_prices")
    if not isinstance(bench, pd.DataFrame) or bench.empty or not isinstance(prices, pd.DataFrame):
        return 1.0, "β = 1 (benchmark not stored)"
    both = pd.concat([prices["adj_close"], bench["adj_close"]], axis=1, join="inner").dropna()
    weekly = both.resample("W-FRI").last().pct_change().dropna().iloc[-260:]
    if len(weekly) < 52:
        return 1.0, "β = 1 (less than one year of common weekly data)"
    raw = float(
        np.cov(weekly.iloc[:, 0], weekly.iloc[:, 1], ddof=1)[0, 1] / weekly.iloc[:, 1].var(ddof=1)
    )
    adjusted = 0.67 * raw + 0.33
    return (
        adjusted,
        f"weekly regression vs {inputs.conventions.get('other_symbol')} ({len(weekly)} w), "
        f"Blume-adjusted (raw {raw:.2f})",
    )


# Damodaran synthetic rating: interest coverage → default spread (large non-financial firms).
SPREADS = [
    (8.5, 0.0059),
    (6.5, 0.0070),
    (5.5, 0.0092),
    (4.25, 0.0107),
    (3.0, 0.0121),
    (2.5, 0.0156),
    (2.25, 0.0200),
    (2.0, 0.0240),
    (1.75, 0.0351),
    (1.5, 0.0421),
    (1.25, 0.0515),
    (0.8, 0.0820),
    (0.65, 0.0864),
    (0.2, 0.1134),
    (-1e9, 0.1500),
]


class WaccParams(BaseModel):
    erp: float = Field(ERP_DEFAULT, ge=0, le=0.15, description="Equity risk premium (D-VA-3)")
    country_premium: float = Field(0.0, ge=0, le=0.15)
    beta: float | None = Field(None, ge=-1, le=4, description="Override the regression β")
    rf: float | None = Field(None, ge=-0.02, le=0.2, description="Override the risk-free rate")
    other: str | None = Field(
        None, max_length=20, pattern=r"^[\w.^=-]+$", description="Benchmark for β"
    )


def wacc_parts(inputs: AnalysisInputs, params: WaccParams) -> dict[str, Any]:
    s = Snapshot(inputs)
    rf = params.rf if params.rf is not None else s.rf
    beta, beta_note = (params.beta, "user override") if params.beta is not None else _beta(inputs)
    ke = rf + beta * params.erp + params.country_premium
    ebit, interest = s.flow_value("operating_income"), s.flow_value("interest_expense")
    coverage = ebit / interest if np.isfinite(interest) and interest > 0 else 100.0
    spread = next(sp for threshold, sp in SPREADS if coverage > threshold)
    kd = rf + spread
    tau = float(tax_rate(s.fy).iloc[-1])
    e, d = s.market_cap, max(s.debt, 0.0)
    wacc = e / (e + d) * ke + d / (e + d) * kd * (1 - tau)
    return {
        "rf": rf,
        "rf_source": "override" if params.rf is not None else s.rf_source,
        "beta": beta,
        "beta_note": beta_note,
        "erp": params.erp,
        "country_premium": params.country_premium,
        "cost_of_equity": ke,
        "interest_coverage": coverage,
        "default_spread": spread,
        "cost_of_debt": kd,
        "tax_rate": tau,
        "equity_weight": e / (e + d),
        "debt_weight": d / (e + d),
        "wacc": wacc,
        "market_cap": e,
        "debt": d,
    }


@analysis(
    id="V-010",
    version="1.0.1",
    title="WACC builder",
    description=(
        "WACC = E/V·k_e + D/V·k_d(1 − τ) at market weights; k_e = rf + β·ERP (+ country "
        "premium), β = 5-year weekly regression vs the benchmark, Blume-adjusted (bottom-up peer β "
        "with J8); k_d = rf + synthetic-rating spread from interest coverage; τ effective, clipped."
    ),
    applies_to=COMPANY,
    params=WaccParams,
    result_kind="table",
    requires={"fundamentals", "prices", "other_prices"},
)
def wacc_builder(inputs: AnalysisInputs, params: WaccParams) -> AnalysisOutput:
    return AnalysisOutput("table", wacc_parts(inputs, params))


# ── V-001 current multiples ──────────────────────────────────────────────────


class NoParams(BaseModel):
    pass


@analysis(
    id="V-001",
    version="1.1.0",
    title="Current multiples",
    description=(
        "P/E, EV/EBITDA, EV/EBIT, EV/Sales, P/FCF, EV/FCF, P/B, P/TBV, P/S, PEG (P/E over 5-year "
        "EPS CAGR in %), earnings / FCF / dividend yields on TTM figures (last FY when quarters "
        "are missing); EV = market cap + debt + leases + preferred + minorities − cash & STI. "
        "Market cap = price × current share count (≤ 400 days old), cross-checked against "
        "net income ÷ diluted EPS (±20 %) for multi-class issuers."
    ),
    applies_to=COMPANY,
    params=NoParams,
    result_kind="table",
    requires=REQUIRES,
)
def current_multiples(inputs: AnalysisInputs, params: NoParams) -> AnalysisOutput:
    s = Snapshot(inputs)
    f = s.flow_value
    ni, ebitda, ebit, revenue, fcf = (
        f("net_income"),
        f("ebitda"),
        f("operating_income"),
        f("revenue"),
        f("fcf"),
    )
    equity = s.stock_value("total_equity")
    tangible = (
        equity
        - (s.stock_value("goodwill") if np.isfinite(s.stock_value("goodwill")) else 0)
        - (s.stock_value("intangibles") if np.isfinite(s.stock_value("intangibles")) else 0)
    )
    dividends = f("dividends_paid") if np.isfinite(f("dividends_paid")) else 0.0
    eps = s.fy["eps_diluted"].dropna()
    eps_cagr = (
        (eps.iloc[-1] / eps.iloc[-6]) ** (1 / 5) - 1
        if len(eps) >= 6 and eps.iloc[-6] > 0 and eps.iloc[-1] > 0
        else None
    )
    pe = s.market_cap / ni if ni > 0 else None

    def ratio(a: float, b: float) -> float | None:
        return _f(a / b) if np.isfinite(b) and b > 0 and np.isfinite(a) else None

    count = s.share_count
    return AnalysisOutput(
        "table",
        {
            "price": s.price,
            "price_date": s.price_date,
            "basis": s.basis,
            "period": s.period,
            "shares": {
                "value": count.value,
                "source": count.source,
                "as_of": count.as_of,
                "outstanding": count.candidates["shares_outstanding"],
                "diluted_wavg": count.candidates["shares_diluted_wavg"],
                "eps_implied": count.eps_implied,
            },
            "market_cap": s.market_cap,
            "enterprise_value": s.ev,
            "net_debt": s.net_debt,
            "currency": s.currency,
            "multiples": {
                "pe": _f(pe),
                "ev_ebitda": ratio(s.ev, ebitda),
                "ev_ebit": ratio(s.ev, ebit),
                "ev_sales": ratio(s.ev, revenue),
                "p_fcf": ratio(s.market_cap, fcf),
                "ev_fcf": ratio(s.ev, fcf),
                "p_b": ratio(s.market_cap, equity),
                "p_tbv": ratio(s.market_cap, tangible),
                "p_s": ratio(s.market_cap, revenue),
                "peg": _f(pe / (eps_cagr * 100)) if pe and eps_cagr and eps_cagr > 0 else None,
            },
            "yields": {
                "earnings": ratio(ni, s.market_cap),
                "fcf": ratio(fcf, s.market_cap),
                "dividend": ratio(dividends, s.market_cap),
            },
        },
        count.notes,
    )


# ── PIT history (V-002, V-020) ───────────────────────────────────────────────

FLOW_ITEMS = [
    "revenue",
    "net_income",
    "operating_income",
    "d_and_a",
    "eps_diluted",
    "cfo",
    "capex",
    "dividends_paid",
]
STOCK_ITEMS = [
    "total_equity",
    "shares_outstanding",
    "shares_diluted_wavg",
    "cash_and_equivalents",
    "short_term_investments",
    "short_term_debt",
    "current_portion_lt_debt",
    "long_term_debt",
    "operating_lease_liabilities",
]


SHARE_ITEMS = ("shares_outstanding", "shares_diluted_wavg")


def split_factor(splits: pd.DataFrame | None, period_end: pd.Timestamp) -> float:
    """Product of split ratios with ex-date after the period: converts an as-reported share
    count (or EPS, inversely) into today's split-adjusted units, matching adjusted prices."""
    if splits is None or splits.empty:
        return 1.0
    later = splits.loc[pd.to_datetime(splits["ex_date"]) > period_end, "ratio"]
    return float(np.prod(later.to_numpy(dtype=float))) if len(later) else 1.0


def pit_ttm(facts_all: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quarterly TTM figures with the date each became known (first filing of its inputs).

    Q4 flows are derived as FY − Q1…Q3 and become known with the 10-K. A rolling TTM is
    known once all four quarters are; instants (equity, debt, shares) at their own filing.
    """
    frame = facts_all.copy()
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["filed_at"] = pd.to_datetime(frame["filed_at"])
    first = frame.sort_values("filed_at").drop_duplicates(
        ["line_item", "period_type", "period_end"], keep="first"
    )
    quarters = first[first["period_type"] == "Q"]
    annual = first[first["period_type"] == "FY"]
    rows = [quarters]
    keys = ["line_item", "fiscal_year"]
    q_index = quarters.groupby(keys)["fiscal_quarter"].agg(set)  # computed once
    q_sums = quarters.groupby(keys)["value"].sum()
    for (code, year), fy in annual.groupby(keys):
        have = q_index.get((code, year), set())
        if 4 in have:
            continue
        if code in STOCK_ITEMS:
            rows.append(fy.assign(period_type="Q", fiscal_quarter=4))
        elif have >= {1, 2, 3} and code in FLOW_ITEMS:
            q4 = float(fy["value"].iloc[0]) - float(q_sums.get((code, year), 0.0))
            rows.append(fy.assign(period_type="Q", fiscal_quarter=4, value=q4))
    q = pd.concat(rows, ignore_index=True).sort_values("period_end")
    if splits is not None and not splits.empty:
        by_end = {end: split_factor(splits, end) for end in q["period_end"].unique()}
        factors = q["period_end"].map(by_end)
        shares_rows = q["line_item"].isin(SHARE_ITEMS)
        q.loc[shares_rows, "value"] = q.loc[shares_rows, "value"] * factors[shares_rows]
        eps_rows = q["line_item"] == "eps_diluted"
        q.loc[eps_rows, "value"] = q.loc[eps_rows, "value"] / factors[eps_rows]
    values = q.pivot_table(index="period_end", columns="line_item", values="value", aggfunc="last")
    known = q.pivot_table(
        index="period_end", columns="line_item", values="filed_at", aggfunc="last"
    )
    out = pd.DataFrame(index=values.index)
    for code in FLOW_ITEMS:
        if code in values:
            out[code] = values[code].rolling(4, min_periods=4).sum()
            seconds = (
                known[code].map(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype(float)
            )
            latest = seconds.rolling(4, min_periods=4).max()  # TTM is known once all 4 quarters are
            out[f"{code}__known"] = pd.to_datetime(latest, unit="s")
    for code in STOCK_ITEMS:
        if code in values:
            out[code] = values[code]
            out[f"{code}__known"] = known[code]
    known_cols = [c for c in out.columns if c.endswith("__known")]
    out["known_at"] = out[known_cols].min(axis=1)  # first item known (row-level indicator only)
    return out.dropna(subset=["known_at"])


def pit_daily(pit: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each date and each item, the latest TTM value whose own filing was public at that
    date (no look-ahead, and a late-tagged item never delays the others)."""
    out = pd.DataFrame(index=dates)
    items = [c for c in pit.columns if not c.endswith("__known") and c != "known_at"]
    for item in items:
        known_col = f"{item}__known"
        if known_col not in pit:
            continue
        series = pit[[item, known_col]].dropna().sort_values(known_col)
        if series.empty:
            out[item] = np.nan
            continue
        positions = (
            np.searchsorted(series[known_col].to_numpy(), dates.to_numpy(), side="right") - 1
        )
        values = series[item].to_numpy()[np.clip(positions, 0, None)]
        out[item] = np.where(positions >= 0, values, np.nan)
        out[known_col] = np.where(
            positions >= 0,
            series[known_col].to_numpy()[np.clip(positions, 0, None)],
            np.datetime64("NaT"),
        )
    return out


def _pit_inputs(inputs: AnalysisInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    facts_all = inputs.data.get("fundamentals_all")
    prices = inputs.data.get("prices")
    if not isinstance(facts_all, pd.DataFrame) or facts_all.empty:
        raise InsufficientDataError("No fundamentals stored: ingest statements first")
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise InsufficientDataError("Valuation history needs stored prices")
    actions = inputs.data.get("actions")
    splits = (
        actions[actions["action"] == "split"]
        if isinstance(actions, pd.DataFrame) and not actions.empty
        else None
    )
    pit = pit_ttm(facts_all, splits)
    if pit.empty:
        raise InsufficientDataError("No quarterly filings: point-in-time TTM history unavailable")
    return pit, prices


class HistoryParams(BaseModel):
    years: int = Field(10, ge=1, le=20)


@analysis(
    id="V-002",
    version="1.0.1",
    title="PIT historical multiples",
    description=(
        "Weekly P/E, P/S, P/B, EV/EBITDA and FCF yield using, at each date, only TTM figures "
        "whose filings were public (first filing date ≤ t); mean ± 1σ and percentiles 10/50/90 "
        "over 5 or 10 years; current percentile."
    ),
    applies_to=COMPANY,
    params=HistoryParams,
    result_kind="series",
    requires={"fundamentals_all", "prices", "actions"},
    budget_ms=6_000,
)
def pit_multiples(inputs: AnalysisInputs, params: HistoryParams) -> AnalysisOutput:
    pit, prices = _pit_inputs(inputs)
    close = prices["close"].resample("W-FRI").last().dropna()
    close = close.loc[close.index >= close.index[-1] - pd.DateOffset(years=params.years)]
    d = pit_daily(pit, pd.DatetimeIndex(close.index))
    shares = d.get("shares_outstanding", pd.Series(np.nan, index=d.index)).fillna(
        d.get("shares_diluted_wavg")
    )
    mcap = close * shares
    debt = sum(
        d[c].fillna(0)
        for c in (
            "short_term_debt",
            "current_portion_lt_debt",
            "long_term_debt",
            "operating_lease_liabilities",
        )
        if c in d
    )
    cash = sum(d[c].fillna(0) for c in ("cash_and_equivalents", "short_term_investments") if c in d)
    ebitda = d.get("operating_income", np.nan) + d.get(
        "d_and_a", pd.Series(0.0, index=d.index)
    ).fillna(0)
    fcf = d.get("cfo", np.nan) - d.get("capex", pd.Series(0.0, index=d.index)).fillna(0)
    series = {
        "pe": div(mcap, d.get("net_income", pd.Series(np.nan, index=d.index))),
        "ps": div(mcap, d.get("revenue", pd.Series(np.nan, index=d.index))),
        "pb": div(mcap, d.get("total_equity", pd.Series(np.nan, index=d.index))),
        "ev_ebitda": div(mcap + debt - cash, ebitda),
        "fcf_yield": div(fcf, mcap),
    }
    stats_out, lines = {}, {}
    for name, s in series.items():
        clean = s.replace([np.inf, -np.inf], np.nan)
        lines[name] = [_f(v) for v in clean]
        valid = clean.dropna()
        if len(valid) < 20:
            continue
        stats_out[name] = {
            "current": _f(valid.iloc[-1]),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "p10": float(valid.quantile(0.1)),
            "p50": float(valid.median()),
            "p90": float(valid.quantile(0.9)),
            "current_percentile": float((valid <= valid.iloc[-1]).mean()),
        }
    if not stats_out:
        raise InsufficientDataError("Too few point-in-time observations for multiple bands")
    return AnalysisOutput(
        "series",
        {
            "dates": [pd.Timestamp(x).date().isoformat() for x in close.index],
            "lines": lines,
            "stats": stats_out,
            "years": params.years,
        },
        (
            "Point-in-time: each figure enters on its first filing date; "
            "later restatements are ignored.",
        ),
    )


@analysis(
    id="V-020",
    version="1.0.1",
    title="Valuation history vs fundamentals",
    description=(
        "Price and point-in-time TTM EPS rebased to 100 over the range, and the log decomposition "
        "of the price return into EPS growth and P/E change, ln(P_T/P_0) = ln(EPS_T/EPS_0) + "
        "ln(PE_T/PE_0), plus the dividend yield earned."
    ),
    applies_to=COMPANY,
    params=HistoryParams,
    result_kind="series",
    requires={"fundamentals_all", "prices", "actions"},
)
def multiple_decomposition(inputs: AnalysisInputs, params: HistoryParams) -> AnalysisOutput:
    pit, prices = _pit_inputs(inputs)
    close = prices["close"].resample("W-FRI").last().dropna()
    close = close.loc[close.index >= close.index[-1] - pd.DateOffset(years=params.years)]
    d = pit_daily(pit, pd.DatetimeIndex(close.index))
    eps = d.get("eps_diluted", pd.Series(np.nan, index=d.index))
    valid = (eps > 0) & close.notna()
    if valid.sum() < 10:
        raise InsufficientDataError("EPS must be positive at both ends for the decomposition")
    first, last = valid.idxmax(), valid[::-1].idxmax()
    price_part = math.log(close[last] / close[first])
    eps_part = math.log(eps[last] / eps[first])
    pe_part = price_part - eps_part
    shares = d.get("shares_outstanding", pd.Series(np.nan, index=d.index)).fillna(
        d.get("shares_diluted_wavg")
    )
    dps = div(d.get("dividends_paid", pd.Series(0.0, index=d.index)).fillna(0), shares)
    years = (last - first).days / 365.25
    dividend_part = float((dps / close).loc[first:last].mean() * years) if years > 0 else 0.0
    return AnalysisOutput(
        "series",
        {
            "dates": [pd.Timestamp(x).date().isoformat() for x in close.index],
            "price_index": [_f(v) for v in close / close[first] * 100],
            "eps_index": [_f(v) for v in eps / eps[first] * 100],
            "decomposition": {
                "price_log_return": price_part,
                "eps_growth": eps_part,
                "multiple_change": pe_part,
                "dividends": dividend_part,
            },
            "from": first.date().isoformat(),
            "to": last.date().isoformat(),
        },
    )


# ── DCF engine (V-011, V-012, V-017, V-016, V-018) ───────────────────────────


class DcfParams(WaccParams):
    growth: float | None = Field(
        None, ge=-0.3, le=0.6, description="Year-1 revenue growth (default: 5y CAGR)"
    )
    margin: float | None = Field(
        None, ge=-0.5, le=0.8, description="EBIT margin (default: 5y median)"
    )
    terminal_growth: float | None = Field(None, ge=-0.02, le=0.06)
    sales_to_capital: float | None = Field(None, gt=0, le=20)
    wacc_override: float | None = Field(None, gt=0.01, le=0.3)
    exit_multiple: float | None = Field(None, gt=0, le=60, description="EV/EBITDA for the exit TV")
    explicit_years: int = Field(5, ge=3, le=10)
    fade_years: int = Field(5, ge=0, le=10)


def dcf_assumptions(inputs: AnalysisInputs, params: DcfParams) -> dict[str, Any]:
    """Seeds per D-VA-4: growth = 5y revenue CAGR (clipped −5…25 %), margin = 5y median EBIT
    margin, sales-to-capital = 5y median revenue / invested capital."""
    s = Snapshot(inputs)
    fy = s.fy
    revenue = fy["revenue"].dropna()
    cagr = (
        (revenue.iloc[-1] / revenue.iloc[-6]) ** 0.2 - 1
        if len(revenue) >= 6 and revenue.iloc[-6] > 0
        else (float(revenue.pct_change().dropna().mean()) if len(revenue) > 1 else 0.03)
    )
    margin_hist = div(fy["operating_income"], fy["revenue"]).dropna().iloc[-5:]
    s2c_hist = div(fy["revenue"], invested_capital(fy)).dropna().iloc[-5:]
    parts = wacc_parts(inputs, params)
    return {
        "snapshot": s,
        "revenue0": float(s.flow_value("revenue")),
        "growth": params.growth if params.growth is not None else float(np.clip(cagr, -0.05, 0.25)),
        "margin": params.margin
        if params.margin is not None
        else float(margin_hist.median())
        if len(margin_hist)
        else 0.1,
        "tax": float(tax_rate(fy).iloc[-1]),
        "sales_to_capital": params.sales_to_capital
        if params.sales_to_capital is not None
        else float(s2c_hist.median())
        if len(s2c_hist) and s2c_hist.median() > 0
        else 1.5,
        "wacc": params.wacc_override if params.wacc_override is not None else parts["wacc"],
        "g": params.terminal_growth if params.terminal_growth is not None else s.g_long,
        "wacc_parts": parts,
    }


def run_dcf(
    a: dict[str, Any],
    params: DcfParams,
    growth: float | None = None,
    margin: float | None = None,
    wacc: float | None = None,
    g: float | None = None,
) -> dict[str, Any]:
    """FCFF projection, growth fading linearly to g over the fade years, mid-year discounting."""
    s: Snapshot = a["snapshot"]
    growth = a["growth"] if growth is None else growth
    margin = a["margin"] if margin is None else margin
    wacc = a["wacc"] if wacc is None else wacc
    g = a["g"] if g is None else g
    if wacc <= g + 0.005:
        raise InsufficientDataError("WACC must exceed terminal growth by at least 0.5 pt")
    years = params.explicit_years + params.fade_years
    rows, revenue, pv = [], a["revenue0"], 0.0
    for t in range(1, years + 1):
        rate = (
            growth
            if t <= params.explicit_years
            else growth + (g - growth) * (t - params.explicit_years) / max(params.fade_years, 1)
        )
        new_revenue = revenue * (1 + rate)
        ebit = new_revenue * margin
        nopat = ebit * (1 - a["tax"])
        reinvestment = (new_revenue - revenue) / a["sales_to_capital"]
        fcff = nopat - reinvestment
        factor = (1 + wacc) ** (t - 0.5)  # mid-year convention (D-VA-1)
        pv += fcff / factor
        rows.append(
            {
                "year": t,
                "growth": rate,
                "revenue": new_revenue,
                "ebit": ebit,
                "nopat": nopat,
                "reinvestment": reinvestment,
                "fcff": fcff,
                "discount": 1 / factor,
            }
        )
        revenue = new_revenue
    last = rows[-1]
    terminal_nopat = last["nopat"] * (1 + g)
    roic_terminal = wacc  # no excess returns in perpetuity: reinvestment rate = g / ROIC
    tv_gordon = terminal_nopat * (1 - g / roic_terminal) / (wacc - g)
    d_and_a_ratio = (
        float(div(s.fy["d_and_a"], s.fy["revenue"]).dropna().iloc[-3:].mean())
        if s.fy["d_and_a"].notna().any()
        else 0.0
    )
    exit_multiple = params.exit_multiple
    tv_exit = (
        (last["ebit"] + last["revenue"] * d_and_a_ratio) * exit_multiple if exit_multiple else None
    )
    discount_tv = (1 + wacc) ** (years - 0.5)
    ev = pv + tv_gordon / discount_tv
    equity = ev - s.net_debt - s.minorities - s.preferred
    per_share = equity / s.shares
    out = {
        "rows": rows,
        "pv_fcff": pv,
        "tv_gordon": tv_gordon,
        "pv_tv": tv_gordon / discount_tv,
        "enterprise_value": ev,
        "equity_value": equity,
        "per_share": per_share,
        "tv_share_of_ev": (tv_gordon / discount_tv) / ev if ev else None,
        "wacc": wacc,
        "g": g,
        "growth": growth,
        "margin": margin,
    }
    if tv_exit is not None:
        ev_exit = pv + tv_exit / discount_tv
        out["per_share_exit"] = (ev_exit - s.net_debt - s.minorities - s.preferred) / s.shares
    return out


@analysis(
    id="V-011",
    version="1.0.1",
    title="DCF",
    description=(
        "FCFF DCF (D-VA-1/4): revenue grows at the seed rate for the explicit years then fades "
        "linearly to g; EBIT margin constant; reinvestment = Δrevenue / sales-to-capital; FCFF = "
        "NOPAT − reinvestment; mid-year discounting at WACC; Gordon TV with ROIC = WACC in "
        "perpetuity (exit-multiple TV alongside when given); EV → equity bridge → per share; "
        "implied multiples check."
    ),
    applies_to=COMPANY,
    params=DcfParams,
    result_kind="table",
    requires={"fundamentals", "prices", "other_prices"},
)
def dcf(inputs: AnalysisInputs, params: DcfParams) -> AnalysisOutput:
    a = dcf_assumptions(inputs, params)
    result = run_dcf(a, params)
    s: Snapshot = a["snapshot"]
    ebitda = s.flow_value("ebitda")
    ni = s.flow_value("net_income")
    return AnalysisOutput(
        "table",
        {
            **{k: v for k, v in result.items()},
            "price": s.price,
            "upside": result["per_share"] / s.price - 1,
            "assumptions": {
                k: a[k]
                for k in ("revenue0", "growth", "margin", "tax", "sales_to_capital", "wacc", "g")
            },
            "wacc_parts": a["wacc_parts"],
            "implied": {
                "ev_ebitda": _f(result["enterprise_value"] / ebitda) if ebitda > 0 else None,
                "pe": _f(result["equity_value"] / ni) if ni > 0 else None,
            },
            "currency": s.currency,
        },
        (
            "Model output from stated assumptions (not a price target).",
            *(
                ("Terminal value is over 75 % of EV: the result rests on perpetuity assumptions.",)
                if (result["tv_share_of_ev"] or 0) > 0.75
                else ()
            ),
        ),
    )


@analysis(
    id="V-012",
    version="1.0.1",
    title="Reverse DCF",
    description=(
        "Revenue growth over the explicit years that makes the DCF value equal the current price "
        "(margin, WACC, g held at their seeds), and the EBIT margin that does so at seed growth."
    ),
    applies_to=COMPANY,
    params=DcfParams,
    result_kind="table",
    requires={"fundamentals", "prices", "other_prices"},
)
def reverse_dcf(inputs: AnalysisInputs, params: DcfParams) -> AnalysisOutput:
    a = dcf_assumptions(inputs, params)
    price = a["snapshot"].price

    def solve(field: str, low: float, high: float) -> float | None:
        def gap(x: float) -> float:
            return float(run_dcf(a, params, **{field: x})["per_share"] - price)

        try:
            return float(optimize.brentq(gap, low, high, xtol=1e-6))
        except ValueError:
            return None

    implied_growth = solve("growth", -0.3, 0.8)
    implied_margin = solve("margin", -0.3, 0.9)
    return AnalysisOutput(
        "table",
        {
            "price": price,
            "implied_growth": implied_growth,
            "implied_margin": implied_margin,
            "seed_growth": a["growth"],
            "seed_margin": a["margin"],
            "wacc": a["wacc"],
            "g": a["g"],
        },
        ()
        if implied_growth is not None
        else ("No growth in −30…80 % reproduces the price at seed margin.",),
    )


class SensitivityParams(DcfParams):
    n_sims: int = Field(2_000, ge=200, le=20_000)
    seed: int = Field(41, ge=0, le=2**32 - 1)


@analysis(
    id="V-017",
    version="1.0.1",
    title="Sensitivity & scenarios",
    description=(
        "Per-share value grids WACC × g (±1 pt by 0.5) and margin × growth (±5 pt); bear/base/"
        "bull scenarios (growth ∓ 5 pt, margin ∓ 3 pt, WACC ± 1 pt); Monte Carlo with growth "
        "triangular(−5, 0, +5 pt), margin normal(σ 2 pt), WACC normal(σ 0.75 pt) → distribution."
    ),
    applies_to=COMPANY,
    params=SensitivityParams,
    result_kind="matrix",
    requires={"fundamentals", "prices", "other_prices"},
    budget_ms=8_000,
)
def sensitivity(inputs: AnalysisInputs, params: SensitivityParams) -> AnalysisOutput:
    a = dcf_assumptions(inputs, params)

    def value(**kw: float) -> float | None:
        try:
            return float(run_dcf(a, params, **kw)["per_share"])
        except InsufficientDataError:
            return None

    waccs = [a["wacc"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    gs = [a["g"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    growths = [a["growth"] + d for d in (-0.05, -0.025, 0, 0.025, 0.05)]
    margins_ = [a["margin"] + d for d in (-0.05, -0.025, 0, 0.025, 0.05)]
    scenarios = {
        "bear": value(growth=a["growth"] - 0.05, margin=a["margin"] - 0.03, wacc=a["wacc"] + 0.01),
        "base": value(),
        "bull": value(growth=a["growth"] + 0.05, margin=a["margin"] + 0.03, wacc=a["wacc"] - 0.01),
    }
    rng = np.random.default_rng(params.seed)
    sims = []
    for _ in range(params.n_sims):
        v = value(
            growth=float(rng.triangular(a["growth"] - 0.05, a["growth"], a["growth"] + 0.05)),
            margin=float(rng.normal(a["margin"], 0.02)),
            wacc=float(rng.normal(a["wacc"], 0.0075)),
        )
        if v is not None and np.isfinite(v):
            sims.append(v)
    sims_arr = np.array(sims)
    price = a["snapshot"].price
    return AnalysisOutput(
        "matrix",
        {
            "price": price,
            "wacc_g": {
                "wacc": waccs,
                "g": gs,
                "values": [[value(wacc=w, g=gg) for gg in gs] for w in waccs],
            },
            "margin_growth": {
                "margin": margins_,
                "growth": growths,
                "values": [[value(margin=m, growth=gr) for gr in growths] for m in margins_],
            },
            "scenarios": scenarios,
            "monte_carlo": {
                "n": len(sims_arr),
                "p5": _f(np.percentile(sims_arr, 5)) if len(sims_arr) else None,
                "p50": _f(np.median(sims_arr)) if len(sims_arr) else None,
                "p95": _f(np.percentile(sims_arr, 95)) if len(sims_arr) else None,
                "prob_above_price": _f((sims_arr > price).mean()) if len(sims_arr) else None,
                "histogram": np.histogram(sims_arr, bins=40)[0].tolist() if len(sims_arr) else [],
                "edges": np.histogram(sims_arr, bins=40)[1].tolist() if len(sims_arr) else [],
            },
        },
    )


@analysis(
    id="V-018",
    version="1.0.1",
    title="Market-implied expectations",
    description=(
        "Iso-value curve: for ROIC from WACC + 1 pt to 60 %, the perpetual growth g such that "
        "EV = NOPAT·(1 − g/ROIC)/(WACC − g) equals the market EV (value-driver formula)."
    ),
    applies_to=COMPANY,
    params=WaccParams,
    result_kind="series",
    requires={"fundamentals", "prices", "other_prices"},
)
def implied_expectations(inputs: AnalysisInputs, params: WaccParams) -> AnalysisOutput:
    parts = wacc_parts(inputs, params)
    s = Snapshot(inputs)
    nopat = s.flow_value("operating_income") * (1 - parts["tax_rate"])
    if not nopat > 0:
        raise InsufficientDataError("Positive NOPAT required for the value-driver formula")
    wacc, ev = parts["wacc"], s.ev
    curve = []
    for roic in np.linspace(wacc + 0.01, 0.6, 40):
        # EV(WACC − g) = NOPAT(1 − g/ROIC)  →  g = (EV·WACC − NOPAT) / (EV − NOPAT/ROIC)
        denominator = ev - nopat / roic
        g = (ev * wacc - nopat) / denominator if denominator else None
        curve.append(
            {"roic": float(roic), "implied_growth": _f(g) if g is not None and g < wacc else None}
        )
    return AnalysisOutput(
        "series", {"wacc": wacc, "nopat": nopat, "enterprise_value": ev, "curve": curve}
    )


# ── V-013 DDM / V-014 residual income / V-015 EPV ────────────────────────────


class DdmParams(WaccParams):
    growth: float | None = Field(None, ge=-0.1, le=0.3, description="Near-term dividend growth")
    terminal_growth: float | None = Field(None, ge=-0.02, le=0.06)
    high_growth_years: int = Field(5, ge=1, le=15)


@analysis(
    id="V-013",
    version="1.0.1",
    title="Dividend discount",
    description=(
        "Gordon P = D1/(k_e − g), two-stage (near-term growth for n years then g) and H-model "
        "P = D0(1 + g_L)/(k_e − g_L) + D0·H·(g_S − g_L)/(k_e − g_L) with H = n/2; payers only. "
        "Near-term growth defaults to the 5-year DPS CAGR (clipped 0–15 %)."
    ),
    applies_to=COMPANY,
    params=DdmParams,
    result_kind="table",
    requires={"fundamentals", "prices", "other_prices"},
)
def dividend_discount(inputs: AnalysisInputs, params: DdmParams) -> AnalysisOutput:
    s = Snapshot(inputs)
    dividends = s.flow_value("dividends_paid")
    if not (np.isfinite(dividends) and dividends > 0):
        raise InsufficientDataError("Not a dividend payer")
    d0 = dividends / s.shares
    ke = wacc_parts(inputs, params)["cost_of_equity"]
    fy_dps = div(
        s.fy["dividends_paid"], s.fy["shares_diluted_wavg"].fillna(s.fy["shares_outstanding"])
    ).dropna()
    hist = (
        (fy_dps.iloc[-1] / fy_dps.iloc[-6]) ** 0.2 - 1
        if len(fy_dps) >= 6 and fy_dps.iloc[-6] > 0
        else 0.05
    )
    gs = params.growth if params.growth is not None else float(np.clip(hist, 0, 0.15))
    gl = params.terminal_growth if params.terminal_growth is not None else s.g_long
    if ke <= gl + 0.005:
        raise InsufficientDataError("Cost of equity must exceed terminal growth")
    gordon = d0 * (1 + gl) / (ke - gl)
    n = params.high_growth_years
    pv, d = 0.0, d0
    for t in range(1, n + 1):
        d *= 1 + gs
        pv += d / (1 + ke) ** t
    two_stage = pv + d * (1 + gl) / (ke - gl) / (1 + ke) ** n
    h_model = d0 * (1 + gl) / (ke - gl) + d0 * (n / 2) * (gs - gl) / (ke - gl)
    return AnalysisOutput(
        "table",
        {
            "dps": d0,
            "cost_of_equity": ke,
            "near_term_growth": gs,
            "terminal_growth": gl,
            "gordon": gordon,
            "two_stage": two_stage,
            "h_model": h_model,
            "price": s.price,
        },
    )


@analysis(
    id="V-014",
    version="1.0.1",
    title="Residual income",
    description=(
        "V0 = B0 + Σ (ROE_t − k_e)·B_{t−1}/(1 + k_e)^t over 10 years, ROE fading linearly from "
        "its 5-year average to k_e, books growing by retained earnings (payout = 5-year average); "
        "the default model for banks and insurers."
    ),
    applies_to=COMPANY,
    params=WaccParams,
    result_kind="table",
    requires={"fundamentals", "prices", "other_prices"},
)
def residual_income(inputs: AnalysisInputs, params: WaccParams) -> AnalysisOutput:
    s = Snapshot(inputs)
    ke = wacc_parts(inputs, params)["cost_of_equity"]
    book0 = s.stock_value("total_equity")
    if not (np.isfinite(book0) and book0 > 0):
        raise InsufficientDataError("Positive book equity required")
    roe_hist = div(s.fy["net_income"], s.fy.avg("total_equity")).dropna().iloc[-5:]
    payout_hist = (
        div(s.fy.z("dividends_paid") + s.fy.z("buybacks"), s.fy["net_income"]).dropna().iloc[-5:]
    )
    roe0 = float(roe_hist.mean()) if len(roe_hist) else ke
    payout = float(np.clip(payout_hist.mean(), 0, 1)) if len(payout_hist) else 0.5
    book, value, rows = book0, book0, []
    for t in range(1, 11):
        roe = roe0 + (ke - roe0) * t / 10
        ri = (roe - ke) * book
        value += ri / (1 + ke) ** t
        rows.append({"year": t, "roe": roe, "book": book, "residual_income": ri})
        book *= 1 + roe * (1 - payout)
    return AnalysisOutput(
        "table",
        {
            "cost_of_equity": ke,
            "roe_start": roe0,
            "payout": payout,
            "book_value": book0,
            "equity_value": value,
            "per_share": value / s.shares,
            "price": s.price,
            "rows": rows,
        },
    )


@analysis(
    id="V-015",
    version="1.0.1",
    title="Earnings power value",
    description=(
        "Greenwald EPV = normalised EBIT(1 − τ)/WACC with EBIT = 5-year average EBIT margin × "
        "current revenue; equity EPV = EPV − net debt; franchise value = market EV − EPV "
        "(> 0: the market pays for growth or a moat)."
    ),
    applies_to=COMPANY,
    params=WaccParams,
    result_kind="table",
    requires={"fundamentals", "prices", "other_prices"},
)
def earnings_power(inputs: AnalysisInputs, params: WaccParams) -> AnalysisOutput:
    s = Snapshot(inputs)
    parts = wacc_parts(inputs, params)
    margin = float(div(s.fy["operating_income"], s.fy["revenue"]).dropna().iloc[-5:].mean())
    ebit = margin * s.flow_value("revenue")
    epv = ebit * (1 - parts["tax_rate"]) / parts["wacc"]
    equity = epv - s.net_debt - s.minorities - s.preferred
    return AnalysisOutput(
        "table",
        {
            "normalised_margin": margin,
            "normalised_ebit": ebit,
            "epv": epv,
            "equity_epv": equity,
            "per_share": equity / s.shares,
            "market_ev": s.ev,
            "franchise_value": s.ev - epv,
            "wacc": parts["wacc"],
            "price": s.price,
        },
    )


# ── V-019 SOTP / V-016 football field ────────────────────────────────────────


class Segment(BaseModel):
    name: str = Field(max_length=40)
    ebitda: float
    multiple: float = Field(gt=0, le=60)


class SotpParams(BaseModel):
    segments: list[Segment] = Field(default_factory=list, max_length=12)


@analysis(
    id="V-019",
    version="1.0.1",
    title="Sum-of-the-parts",
    description=(
        "Σ segment EBITDA × user multiple − net debt − minorities, per share (segments entered "
        "manually until XBRL segments, F-009)."
    ),
    applies_to=COMPANY,
    params=SotpParams,
    result_kind="table",
    requires=REQUIRES,
)
def sum_of_parts(inputs: AnalysisInputs, params: SotpParams) -> AnalysisOutput:
    if not params.segments:
        raise InsufficientDataError(
            "Enter segments (name, EBITDA, multiple): XBRL segments arrive with F-009"
        )
    s = Snapshot(inputs)
    parts = [{"name": seg.name, "value": seg.ebitda * seg.multiple} for seg in params.segments]
    ev = float(sum(seg.ebitda * seg.multiple for seg in params.segments))
    equity = ev - s.net_debt - s.minorities - s.preferred
    return AnalysisOutput(
        "table",
        {"parts": parts, "enterprise_value": ev, "per_share": equity / s.shares, "price": s.price},
    )


@analysis(
    id="V-016",
    version="1.1.1",
    title="Football field",
    description=(
        "Per-share low–high ranges vs the current price (D-VA-6): DCF bear–bull, own-history "
        "P/E and EV/EBITDA (p10–p90 × current TTM), DDM (payers), residual income, EPV, 52-week "
        "range, peer multiples (PE-004 implied values from peer-median P/E, EV/EBITDA, EV/Sales: "
        "min–max). Analyst targets are not used."
    ),
    applies_to=COMPANY,
    params=DcfParams,
    result_kind="table",
    requires={"fundamentals", "fundamentals_all", "prices", "other_prices", "actions", "peers"},
    budget_ms=10_000,
)
def football_field(inputs: AnalysisInputs, params: DcfParams) -> AnalysisOutput:
    s = Snapshot(inputs)
    bars: list[dict[str, Any]] = []

    def add(name: str, low: float | None, high: float | None, note: str = "") -> None:
        if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
            bars.append(
                {"method": name, "low": min(low, high), "high": max(low, high), "note": note}
            )

    try:
        a = dcf_assumptions(inputs, params)
        bear = run_dcf(
            a, params, growth=a["growth"] - 0.05, margin=a["margin"] - 0.03, wacc=a["wacc"] + 0.01
        )["per_share"]
        bull = run_dcf(
            a, params, growth=a["growth"] + 0.05, margin=a["margin"] + 0.03, wacc=a["wacc"] - 0.01
        )["per_share"]
        add("DCF (bear–bull)", bear, bull)
    except InsufficientDataError as exc:
        bars.append({"method": "DCF", "low": None, "high": None, "note": str(exc)})
    try:
        hist = pit_multiples(inputs, HistoryParams.model_validate({"years": 10})).payload["stats"]
        ni, ebitda = s.flow_value("net_income"), s.flow_value("ebitda")
        if "pe" in hist and ni > 0:
            add(
                "Own-history P/E (p10–p90)",
                hist["pe"]["p10"] * ni / s.shares,
                hist["pe"]["p90"] * ni / s.shares,
            )
        if "ev_ebitda" in hist and ebitda > 0:

            def to_share(multiple: float) -> float:
                return (multiple * ebitda - s.net_debt - s.minorities - s.preferred) / s.shares

            add(
                "Own-history EV/EBITDA (p10–p90)",
                to_share(hist["ev_ebitda"]["p10"]),
                to_share(hist["ev_ebitda"]["p90"]),
            )
    except InsufficientDataError as exc:
        bars.append(
            {"method": "Own-history multiples", "low": None, "high": None, "note": str(exc)}
        )
    for name, fn, args in (
        ("DDM (Gordon–2-stage)", dividend_discount, DdmParams.model_validate({})),
        ("Residual income", residual_income, WaccParams.model_validate({})),
        ("Earnings power value", earnings_power, WaccParams.model_validate({})),
    ):
        try:
            out = fn(inputs, args).payload  # type: ignore[arg-type]
            if name.startswith("DDM"):
                add(name, out["gordon"], out["two_stage"])
            else:
                add(
                    name,
                    out["per_share"] * 0.9,
                    out["per_share"] * 1.1,
                    "±10 % around the point value",
                )
        except InsufficientDataError as exc:
            bars.append({"method": name, "low": None, "high": None, "note": str(exc)})
    prices = inputs.data["prices"]
    year = prices["close"].loc[prices.index >= prices.index[-1] - pd.Timedelta(days=365)]
    add("52-week range", float(year.min()), float(year.max()))
    from fda_core.analyses.peers import RelativeParams, relative_valuation

    try:
        implied = relative_valuation(inputs, RelativeParams()).payload["implied_per_share"]
        values = [
            x
            for k in ("pe", "ev_ebitda", "ev_sales")
            if (x := implied.get(k)) is not None and np.isfinite(x) and x > 0
        ]
        if values:
            add(
                "Peer multiples", min(values), max(values), "peer medians: P/E, EV/EBITDA, EV/Sales"
            )
        else:
            bars.append(
                {
                    "method": "Peer multiples",
                    "low": None,
                    "high": None,
                    "note": "no usable peer multiple",
                }
            )
    except InsufficientDataError as exc:
        bars.append({"method": "Peer multiples", "low": None, "high": None, "note": str(exc)})
    return AnalysisOutput(
        "table",
        {"price": s.price, "currency": s.currency, "bars": bars},
        ("Ranges are model outputs from stated assumptions, not price targets.",),
    )
