"""Financial statements (ANALYTICS_CATALOG F-001…F-008, F-021, F-022; module M08).

Input `data["fundamentals"]`: point-in-time long facts (one row per line item × period type ×
period end, latest filing known at the as-of date) with columns line_item, period_type,
fiscal_year, fiscal_quarter, period_end, value, unit, filed_at, source, form, is_restated.
Derived lines (EBITDA, FCF, Q4 = FY − Q1…Q3, TTM) are computed here, never stored.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.registry import (
    AnalysisInputs,
    AnalysisOutput,
    InsufficientDataError,
    NotApplicableError,
    SubjectKind,
    analysis,
)

COMPANY: set[SubjectKind] = {"asset"}
Basis = Literal["FY", "Q", "TTM"]

# Canonical order and metadata (mirrors fda_data/mapping/line_items.yaml).
STATEMENTS: dict[str, list[tuple[str, str, int]]] = {
    "IS": [
        ("revenue", "Revenue", 1),
        ("cost_of_revenue", "Cost of revenue", -1),
        ("gross_profit", "Gross profit", 1),
        ("rnd", "Research & development", -1),
        ("sga", "Selling, general & admin.", -1),
        ("other_opex", "Other operating expenses", -1),
        ("operating_expenses", "Operating expenses", -1),
        ("operating_income", "Operating income (EBIT)", 1),
        ("d_and_a", "Depreciation & amortization", 1),
        ("ebitda", "EBITDA", 1),
        ("interest_expense", "Interest expense", -1),
        ("interest_income", "Interest income", 1),
        ("other_non_operating", "Other non-operating", 1),
        ("pretax_income", "Pre-tax income", 1),
        ("income_tax", "Income tax", -1),
        ("net_income", "Net income", 1),
        ("minority_interest_is", "Minority interest", 1),
        ("net_income_common", "Net income to common", 1),
        ("eps_basic", "EPS basic", 1),
        ("eps_diluted", "EPS diluted", 1),
        ("shares_diluted_wavg", "Shares diluted (w. avg)", 1),
        ("sbc", "Stock-based compensation", 1),
    ],
    "BS": [
        ("cash_and_equivalents", "Cash & equivalents", 1),
        ("short_term_investments", "Short-term investments", 1),
        ("receivables", "Receivables", 1),
        ("inventory", "Inventory", 1),
        ("other_current_assets", "Other current assets", 1),
        ("total_current_assets", "Total current assets", 1),
        ("ppe_net", "PP&E (net)", 1),
        ("goodwill", "Goodwill", 1),
        ("intangibles", "Intangibles", 1),
        ("other_non_current_assets", "Other non-current assets", 1),
        ("total_assets", "Total assets", 1),
        ("accounts_payable", "Accounts payable", 1),
        ("short_term_debt", "Short-term debt", 1),
        ("current_portion_lt_debt", "Current portion of LT debt", 1),
        ("other_current_liabilities", "Other current liabilities", 1),
        ("total_current_liabilities", "Total current liabilities", 1),
        ("long_term_debt", "Long-term debt", 1),
        ("operating_lease_liabilities", "Operating lease liabilities", 1),
        ("other_non_current_liabilities", "Other non-current liabilities", 1),
        ("total_liabilities", "Total liabilities", 1),
        ("common_equity", "Common equity", 1),
        ("retained_earnings", "Retained earnings", 1),
        ("minority_interest_bs", "Minority interest", 1),
        ("total_equity", "Total equity", 1),
        ("total_liabilities_and_equity", "Total liabilities & equity", 1),
        ("net_debt", "Net debt", 1),
        ("shares_outstanding", "Shares outstanding", 1),
    ],
    "CF": [
        ("cfo", "Cash from operations", 1),
        ("capex", "Capital expenditure", -1),
        ("fcf", "Free cash flow", 1),
        ("sbc", "Stock-based compensation", 1),
        ("fcf_minus_sbc", "FCF − SBC", 1),
        ("acquisitions", "Acquisitions", -1),
        ("cfi", "Cash from investing", 1),
        ("dividends_paid", "Dividends paid", -1),
        ("buybacks", "Share buybacks", -1),
        ("share_issuance", "Share issuance", 1),
        ("debt_issued", "Debt issued", 1),
        ("debt_repaid", "Debt repaid", -1),
        ("cff", "Cash from financing", 1),
        ("fx_effect", "FX effect", 1),
        ("net_change_cash", "Net change in cash", 1),
    ],
}
STOCKS = (
    {code for code, _, _ in STATEMENTS["BS"]}
    | {f"debt_maturity_y{i}" for i in range(1, 6)}
    | {"debt_maturity_after_y5", "employees"}
)
PER_SHARE = {"eps_basic", "eps_diluted"}
COUNTS = {"shares_basic_wavg", "shares_diluted_wavg", "shares_outstanding", "employees"}
DERIVED = {"ebitda", "fcf", "fcf_minus_sbc", "net_debt", "total_liabilities_and_equity"}


def _facts(inputs: AnalysisInputs) -> pd.DataFrame:
    if inputs.conventions.get("asset_class", "equity") != "equity":
        raise NotApplicableError(
            "Financial statements apply to companies (equity), not this asset type"
        )
    facts = inputs.data.get("fundamentals")
    if not isinstance(facts, pd.DataFrame) or facts.empty:
        raise InsufficientDataError("No fundamentals stored: ingest statements first")
    frame = facts.copy()
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.date
    return frame


def _label(period_type: str, year: int, quarter: Any) -> str:
    if period_type == "FY":
        return f"FY{year}"
    return f"FY{year} Q{int(quarter)}"


def build_periods(facts: pd.DataFrame, basis: Basis) -> pd.DataFrame:
    """Wide table: index = line items, columns = periods (oldest → newest), with derived lines.

    Q adds Q4 = FY − (Q1 + Q2 + Q3) for flows when the 10-K only reports the year; TTM sums the
    last four quarters for flows and takes the latest instant for stocks and counts.
    """
    if basis in {"Q", "TTM"}:
        quarters = facts[facts["period_type"] == "Q"]
        annual = facts[facts["period_type"] == "FY"]
        rows = [quarters]
        # Quarter numbers and Q1–Q3 sums per (line item, fiscal year), computed once.
        q_index = quarters.groupby(["line_item", "fiscal_year"])["fiscal_quarter"].agg(set)
        q_sums = (
            quarters[quarters["fiscal_quarter"].isin([1, 2, 3])]
            .groupby(["line_item", "fiscal_year"])["value"]
            .sum()
        )
        for (code, year), fy in annual.groupby(["line_item", "fiscal_year"]):
            have = q_index.get((code, year), set())
            if code in STOCKS or code in COUNTS:
                if 4 not in have:
                    rows.append(fy.assign(period_type="Q", fiscal_quarter=4))
            elif have >= {1, 2, 3} and 4 not in have:
                if code in PER_SHARE:
                    continue  # EPS do not add up across quarters
                q4 = float(fy["value"].iloc[0]) - float(q_sums.get((code, year), 0.0))
                rows.append(
                    fy.assign(period_type="Q", fiscal_quarter=4, value=q4, source="derived")
                )
        base = pd.concat(rows, ignore_index=True)
        base["period"] = [
            _label("Q", y, q)
            for y, q in zip(base["fiscal_year"], base["fiscal_quarter"], strict=True)
        ]
    else:
        base = facts[facts["period_type"] == "FY"].copy()
        base["period"] = [f"FY{y}" for y in base["fiscal_year"]]
    if base.empty:
        raise InsufficientDataError(f"No {basis} facts stored")
    base = base.sort_values("period_end")
    wide = base.pivot_table(index="line_item", columns="period", values="value", aggfunc="last")
    order = base.drop_duplicates("period", keep="last").sort_values("period_end")["period"].tolist()
    wide = wide[order]
    if basis == "TTM":
        flows = wide.index.difference(list(STOCKS | COUNTS | PER_SHARE))
        ttm = wide.copy()
        ttm.loc[flows] = wide.loc[flows].T.rolling(4, min_periods=4).sum().T
        if "eps_diluted" in wide.index:
            ttm.loc["eps_diluted"] = wide.loc["eps_diluted"].T.rolling(4, min_periods=4).sum().T
        wide = ttm.iloc[:, 3:]
        wide.columns = [f"TTM {c}" for c in wide.columns]
    return add_derived(wide)


def add_derived(wide: pd.DataFrame) -> pd.DataFrame:
    def get(code: str) -> pd.Series:
        return wide.loc[code] if code in wide.index else pd.Series(np.nan, index=wide.columns)

    if "gross_profit" not in wide.index or wide.loc["gross_profit"].isna().all():
        wide.loc["gross_profit"] = get("revenue") - get("cost_of_revenue")
    wide.loc["ebitda"] = get("operating_income") + get("d_and_a")
    wide.loc["fcf"] = get("cfo") - get("capex")
    wide.loc["fcf_minus_sbc"] = get("fcf") - get("sbc")
    debt = (
        get("short_term_debt").fillna(0)
        + get("current_portion_lt_debt").fillna(0)
        + get("long_term_debt").fillna(0)
    )
    cash = get("cash_and_equivalents").fillna(0) + get("short_term_investments").fillna(0)
    wide.loc["net_debt"] = (debt - cash).where(get("total_assets").notna())
    if (
        "total_liabilities_and_equity" not in wide.index
        or wide.loc["total_liabilities_and_equity"].isna().all()
    ):
        wide.loc["total_liabilities_and_equity"] = get("total_liabilities") + get("total_equity")
    if "total_liabilities" not in wide.index or wide.loc["total_liabilities"].isna().all():
        wide.loc["total_liabilities"] = get("total_liabilities_and_equity") - get("total_equity")
    return wide


def _clean(values: Any) -> list[float | None]:
    return [
        None if v is None or not np.isfinite(v) else float(v)
        for v in np.asarray(values, dtype=float)
    ]


# ── F-001 / F-002 / F-003 statement tables ───────────────────────────────────


class StatementParams(BaseModel):
    basis: Basis = "FY"
    periods: int = Field(10, ge=1, le=40)
    common_size: bool = False


def _statement(inputs: AnalysisInputs, params: StatementParams, statement: str) -> AnalysisOutput:
    facts = _facts(inputs)
    wide = build_periods(facts, params.basis).iloc[:, -params.periods :]
    base_code = {"IS": "revenue", "BS": "total_assets", "CF": "cfo"}[statement]
    base = wide.loc[base_code] if base_code in wide.index else None
    # Flag a line only when a period on screen was changed by a later filing.
    shown_years = {
        int(str(p).removeprefix("TTM ").removeprefix("FY").split(" ")[0]) for p in wide.columns
    }
    flagged = facts[facts["is_restated"].astype(bool)] if "is_restated" in facts else facts.iloc[:0]
    restated = set(flagged.loc[flagged["fiscal_year"].isin(shown_years), "line_item"])
    sources = (
        facts.drop_duplicates("line_item", keep="last").set_index("line_item")["source"].to_dict()
    )
    rows = []
    for code, label, sign in STATEMENTS[statement]:
        if code not in wide.index or wide.loc[code].isna().all():
            continue
        values = wide.loc[code]
        lag = 4 if params.basis == "Q" else 1
        yoy = values / values.shift(lag) - 1
        row: dict[str, Any] = {
            "code": code,
            "label": label,
            "sign": sign,
            "values": _clean(values),
            "yoy": _clean(yoy.where(values.shift(lag).abs() > 0)),
            "derived": code in DERIVED,
            "restated": code in restated,
            "source": "engine" if code in DERIVED else sources.get(code),
            "kind": "per_share" if code in PER_SHARE else "count" if code in COUNTS else "amount",
        }
        if params.common_size and base is not None and code not in PER_SHARE | COUNTS:
            row["common_size"] = _clean(values / base)
        rows.append(row)
    unit = facts.loc[facts["unit"].astype(str).str.len() == 3, "unit"]
    return AnalysisOutput(
        "table",
        {
            "statement": statement,
            "basis": params.basis,
            "periods": list(wide.columns),
            "currency": unit.iloc[0] if len(unit) else None,
            "rows": rows,
            "sources": sorted(set(facts["source"])),
        },
        ()
        if params.basis != "Q"
        else ("Q4 flows are derived as FY − (Q1 + Q2 + Q3) when not reported.",),
    )


def _register_statement(analysis_id: str, title: str, statement: str, description: str) -> None:
    def fn(inputs: AnalysisInputs, params: StatementParams) -> AnalysisOutput:
        return _statement(inputs, params, statement)

    analysis(
        id=analysis_id,
        version="1.0.0",
        title=title,
        description=description,
        applies_to=COMPANY,
        params=StatementParams,
        result_kind="table",
        requires={"fundamentals"},
    )(fn)


_register_statement(
    "F-001",
    "Income statement",
    "IS",
    "Canonical income statement, FY / quarterly / TTM, YoY (QoQ-year for quarters), optional "
    "common-size % of revenue, EBITDA = EBIT + D&A derived; restated items flagged.",
)
_register_statement(
    "F-002",
    "Balance sheet",
    "BS",
    "Canonical balance sheet at period ends, common-size % of total assets, YoY, net debt "
    "derived (debt − cash − short-term investments).",
)
_register_statement(
    "F-003",
    "Cash-flow statement",
    "CF",
    "Cash flows from operations, investing and financing with derived FCF = CFO − capex and "
    "FCF − SBC; outflows are positive amounts shown with a minus sign.",
)


# ── F-004 Sankey / F-005 waterfall / F-021 cash bridge ───────────────────────


class PeriodParams(BaseModel):
    basis: Literal["FY", "TTM"] = "FY"
    period: str | None = Field(None, max_length=16, description="Period label; default latest")


def _column(inputs: AnalysisInputs, params: PeriodParams) -> tuple[str, pd.Series]:
    wide = build_periods(_facts(inputs), params.basis)
    column = params.period if params.period in wide.columns else wide.columns[-1]
    return str(column), wide[column]


def _value(column: pd.Series, code: str) -> float:
    value = column.get(code)
    return 0.0 if value is None or not np.isfinite(value) else float(value)


@analysis(
    id="F-004",
    version="1.0.0",
    title="Income Sankey",
    description=(
        "Revenue → cost of revenue / gross profit → operating expenses (R&D, SG&A, other) / "
        "operating income → interest, other, tax / net income, as nodes and links for a period."
    ),
    applies_to=COMPANY,
    params=PeriodParams,
    result_kind="matrix",
    requires={"fundamentals"},
)
def income_sankey(inputs: AnalysisInputs, params: PeriodParams) -> AnalysisOutput:
    period, col = _column(inputs, params)
    revenue, cogs = _value(col, "revenue"), _value(col, "cost_of_revenue")
    gross = _value(col, "gross_profit") or revenue - cogs
    ebit = _value(col, "operating_income")
    rnd, sga = _value(col, "rnd"), _value(col, "sga")
    other_opex = max(gross - ebit - rnd - sga, 0.0)
    tax, net = _value(col, "income_tax"), _value(col, "net_income")
    other_below = max(ebit - tax - net, 0.0)
    if revenue <= 0:
        raise InsufficientDataError("No revenue for this period")
    links = [
        ("Revenue", "Cost of revenue", cogs),
        ("Revenue", "Gross profit", gross),
        ("Gross profit", "R&D", rnd),
        ("Gross profit", "SG&A", sga),
        ("Gross profit", "Other opex", other_opex),
        ("Gross profit", "Operating income", ebit),
        ("Operating income", "Income tax", tax),
        ("Operating income", "Interest & other", other_below),
        ("Operating income", "Net income", net),
    ]
    links = [(s, t, v) for s, t, v in links if v > 0]
    nodes = sorted({n for s, t, _ in links for n in (s, t)})
    return AnalysisOutput(
        "matrix",
        {
            "period": period,
            "nodes": [{"name": n} for n in nodes],
            "links": [{"source": s, "target": t, "value": v} for s, t, v in links],
            "margins": {
                "gross": gross / revenue,
                "operating": ebit / revenue,
                "net": net / revenue,
            },
        },
    )


def _waterfall(steps: list[tuple[str, float, bool]]) -> list[dict[str, Any]]:
    """Steps (label, amount, is_total) → bars with base/height for a floating waterfall."""
    out, running = [], 0.0
    for label, amount, total in steps:
        if total:
            out.append({"label": label, "value": amount, "base": 0.0, "total": True})
            running = amount
        else:
            out.append({"label": label, "value": amount, "base": running, "total": False})
            running += amount
    return out


@analysis(
    id="F-005",
    version="1.0.0",
    title="Profit waterfall",
    description=(
        "Revenue − cost of revenue = gross profit − operating expenses = EBIT − interest/other "
        "− tax = net income."
    ),
    applies_to=COMPANY,
    params=PeriodParams,
    result_kind="series",
    requires={"fundamentals"},
)
def profit_waterfall(inputs: AnalysisInputs, params: PeriodParams) -> AnalysisOutput:
    period, col = _column(inputs, params)
    revenue = _value(col, "revenue")
    gross = _value(col, "gross_profit") or revenue - _value(col, "cost_of_revenue")
    ebit, pretax, net = (
        _value(col, "operating_income"),
        _value(col, "pretax_income"),
        _value(col, "net_income"),
    )
    steps = [
        ("Revenue", revenue, True),
        ("Cost of revenue", gross - revenue, False),
        ("Gross profit", gross, True),
        ("Operating expenses", ebit - gross, False),
        ("Operating income", ebit, True),
        ("Interest & other", pretax - ebit, False),
        ("Pre-tax income", pretax, True),
        ("Income tax", net - pretax, False),
        ("Net income", net, True),
    ]
    return AnalysisOutput("series", {"period": period, "bars": _waterfall(steps)})


@analysis(
    id="F-021",
    version="1.0.0",
    title="Cash-flow bridge",
    description=(
        "EBITDA → taxes → working capital & other (CFO − EBITDA + tax) → CFO → capex → FCF → "
        "dividends → buybacks → acquisitions → change in net debt (residual, sign: increase "
        "in net debt negative)."
    ),
    applies_to=COMPANY,
    params=PeriodParams,
    result_kind="series",
    requires={"fundamentals"},
)
def cash_bridge(inputs: AnalysisInputs, params: PeriodParams) -> AnalysisOutput:
    period, col = _column(inputs, params)
    ebitda, tax, cfo = _value(col, "ebitda"), _value(col, "income_tax"), _value(col, "cfo")
    capex, dividends = _value(col, "capex"), _value(col, "dividends_paid")
    buybacks, acquisitions = _value(col, "buybacks"), _value(col, "acquisitions")
    fcf = cfo - capex
    after = fcf - dividends - buybacks - acquisitions
    steps = [
        ("EBITDA", ebitda, True),
        ("Taxes", -tax, False),
        ("Working capital & other", cfo - ebitda + tax, False),
        ("CFO", cfo, True),
        ("Capex", -capex, False),
        ("Free cash flow", fcf, True),
        ("Dividends", -dividends, False),
        ("Buybacks", -buybacks, False),
        ("Acquisitions", -acquisitions, False),
        ("Cash left (− = funded by debt/cash)", after, True),
    ]
    return AnalysisOutput("series", {"period": period, "bars": _waterfall(steps)})


# ── F-006 balance structure / F-007 debt ladder / F-008 multi-year ───────────


class YearsParams(BaseModel):
    periods: int = Field(10, ge=2, le=30)


@analysis(
    id="F-006",
    version="1.0.0",
    title="Balance structure",
    description=(
        "Assets (cash, receivables, inventory, PP&E, goodwill & intangibles, other) vs "
        "liabilities and equity per fiscal year."
    ),
    applies_to=COMPANY,
    params=YearsParams,
    result_kind="matrix",
    requires={"fundamentals"},
)
def balance_structure(inputs: AnalysisInputs, params: YearsParams) -> AnalysisOutput:
    wide = build_periods(_facts(inputs), "FY").iloc[:, -params.periods :]

    def row(code: str) -> pd.Series:
        return (
            wide.loc[code].fillna(0) if code in wide.index else pd.Series(0.0, index=wide.columns)
        )

    assets = row("total_assets")
    parts_assets = {
        "cash & investments": row("cash_and_equivalents") + row("short_term_investments"),
        "receivables": row("receivables"),
        "inventory": row("inventory"),
        "PP&E": row("ppe_net"),
        "goodwill & intangibles": row("goodwill") + row("intangibles"),
    }
    parts_assets["other"] = assets - sum(parts_assets.values())
    debt = row("short_term_debt") + row("current_portion_lt_debt") + row("long_term_debt")
    equity = row("total_equity")
    parts_liab = {"debt": debt, "other liabilities": assets - equity - debt, "equity": equity}
    return AnalysisOutput(
        "matrix",
        {
            "periods": list(wide.columns),
            "assets": {k: _clean(v) for k, v in parts_assets.items()},
            "liabilities_equity": {k: _clean(v) for k, v in parts_liab.items()},
        },
    )


@analysis(
    id="F-007",
    version="1.0.0",
    title="Debt maturity ladder",
    description=(
        "Principal due in years 1–5 and after (XBRL maturities, latest FY) against cash and FCF."
    ),
    applies_to=COMPANY,
    params=PeriodParams,
    result_kind="table",
    requires={"fundamentals"},
)
def debt_ladder(inputs: AnalysisInputs, params: PeriodParams) -> AnalysisOutput:
    period, col = _column(inputs, params)
    codes = [f"debt_maturity_y{i}" for i in range(1, 6)] + ["debt_maturity_after_y5"]
    ladder = [{"bucket": c.replace("debt_maturity_", ""), "amount": _value(col, c)} for c in codes]
    if not any(r["amount"] for r in ladder):
        raise InsufficientDataError("The filer does not tag debt maturities (XBRL) for this period")
    return AnalysisOutput(
        "table",
        {
            "period": period,
            "ladder": ladder,
            "cash": _value(col, "cash_and_equivalents") + _value(col, "short_term_investments"),
            "fcf": _value(col, "fcf"),
            "total": sum(r["amount"] for r in ladder),
        },
    )


class MultiYearParams(BaseModel):
    items: list[str] = Field(
        default=["revenue", "operating_income", "net_income", "fcf"], min_length=1, max_length=6
    )
    basis: Basis = "FY"
    periods: int = Field(10, ge=2, le=40)


@analysis(
    id="F-008",
    version="1.0.0",
    title="Multi-year fundamentals chart",
    description="Chosen line items as bars per period with gross, operating, net and FCF margins.",
    applies_to=COMPANY,
    params=MultiYearParams,
    result_kind="series",
    requires={"fundamentals"},
)
def multi_year(inputs: AnalysisInputs, params: MultiYearParams) -> AnalysisOutput:
    wide = build_periods(_facts(inputs), params.basis).iloc[:, -params.periods :]
    revenue = wide.loc["revenue"] if "revenue" in wide.index else None
    bars = {code: _clean(wide.loc[code]) for code in params.items if code in wide.index}
    margins = {}
    if revenue is not None:
        for name, code in (
            ("gross", "gross_profit"),
            ("operating", "operating_income"),
            ("net", "net_income"),
            ("fcf", "fcf"),
        ):
            if code in wide.index:
                margins[name] = _clean(wide.loc[code] / revenue)
    return AnalysisOutput(
        "series", {"periods": list(wide.columns), "bars": bars, "margins": margins}
    )


# ── F-022 statement quality checks ───────────────────────────────────────────


@analysis(
    id="F-022",
    version="1.0.0",
    title="Statement quality checks",
    description=(
        "Reconciliation per fiscal year: assets = liabilities + equity (±0.5 %), gross profit = "
        "revenue − cost of revenue (±0.5 %), Q1…Q4 flows vs FY (±1 %); restated items; coverage "
        "matrix of statements × fiscal years × source; estimated-PIT flag for providers without "
        "filing dates."
    ),
    applies_to=COMPANY,
    params=YearsParams,
    result_kind="matrix",
    requires={"fundamentals"},
)
def statement_quality(inputs: AnalysisInputs, params: YearsParams) -> AnalysisOutput:
    facts = _facts(inputs)
    wide = build_periods(facts, "FY").iloc[:, -params.periods :]
    checks = []
    for period in wide.columns:
        col = wide[period]
        row: dict[str, Any] = {"period": period}
        a, liab, eq = col.get("total_assets"), col.get("total_liabilities"), col.get("total_equity")
        if all(v is not None and np.isfinite(v) for v in (a, liab, eq)) and a:
            row["balance_identity_gap"] = float((a - liab - eq) / a)
            row["balance_ok"] = abs(row["balance_identity_gap"]) <= 0.005
        rev, cogs, gp = col.get("revenue"), col.get("cost_of_revenue"), col.get("gross_profit")
        if all(v is not None and np.isfinite(v) for v in (rev, cogs, gp)) and rev:
            row["gross_profit_gap"] = float((rev - cogs - gp) / rev)
            row["gross_profit_ok"] = abs(row["gross_profit_gap"]) <= 0.005
        year = int(period.removeprefix("FY"))
        quarters = facts[
            (facts["period_type"] == "Q")
            & (facts["line_item"] == "revenue")
            & (facts["fiscal_year"] == year)
        ]
        if len(set(quarters["fiscal_quarter"])) == 4 and rev:
            row["quarters_vs_fy_gap"] = float((quarters["value"].sum() - rev) / rev)
            row["quarters_ok"] = abs(row["quarters_vs_fy_gap"]) <= 0.01
        checks.append(row)
    coverage = []
    statement_of = {code: st for st, rows in STATEMENTS.items() for code, _, _ in rows}
    fy = facts[facts["period_type"] == "FY"].assign(
        statement=lambda f: f["line_item"].map(statement_of)
    )
    for (statement, year, source), group in fy.dropna(subset=["statement"]).groupby(
        ["statement", "fiscal_year", "source"]
    ):
        coverage.append(
            {
                "statement": statement,
                "fiscal_year": int(year),
                "source": source,
                "items": int(group["line_item"].nunique()),
            }
        )
    restated = facts[facts["is_restated"].astype(bool)][
        ["line_item", "period_type", "period_end", "value", "filed_at"]
    ]
    return AnalysisOutput(
        "matrix",
        {
            "checks": checks,
            "coverage": coverage,
            "restated": [
                {**r, "period_end": str(r["period_end"]), "filed_at": str(r["filed_at"])}
                for r in restated.tail(50).to_dict("records")
            ],
            "estimated_pit": bool(facts.get("estimated_pit", pd.Series(False)).astype(bool).any()),
            "sources": sorted(set(facts["source"])),
        },
        (
            (
                "Filing dates estimated (period end + 75 d FY / 45 d Q): "
                "point-in-time is approximate.",
            )
            if facts.get("estimated_pit", pd.Series(False)).astype(bool).any()
            else ()
        ),
    )
