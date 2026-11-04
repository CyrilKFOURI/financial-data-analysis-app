"""Ratios, financial health & forensics (ANALYTICS_CATALOG F-010…F-025, S-001…S-010;
modules M09–M10). Built on the fiscal-year table of `statements.build_periods`.

Conventions (DECISIONS D-FU-2…D-FU-7): averages use (t, t−1) balances; ROIC = EBIT(1 − τ) /
avg(debt + leases + equity − cash), τ = effective tax rate clipped to [0, 35 %]; operating
leases count as debt; SBC shown separately. Ratios are null (with a reason) when an input is
missing or a denominator is ≤ 0 — never a misleading number.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from fda_core.analyses.statements import COMPANY, _facts, build_periods
from fda_core.registry import (
    AnalysisInputs,
    AnalysisOutput,
    InsufficientDataError,
    NotApplicableError,
    analysis,
)

DAYS = 365.0


class Book:
    """Fiscal-year accessor: `b["revenue"]` is a Series over periods (NaN when missing)."""

    def __init__(self, wide: pd.DataFrame) -> None:
        self.wide = wide
        self.periods = list(wide.columns)

    def __getitem__(self, code: str) -> pd.Series:
        if code in self.wide.index:
            return self.wide.loc[code].astype(float)
        return pd.Series(np.nan, index=self.periods, dtype=float)

    def z(self, code: str) -> pd.Series:
        """Missing treated as 0 (for optional components such as leases or STI)."""
        return self[code].fillna(0.0)

    def avg(self, code: str) -> pd.Series:
        s = self[code]
        return (s + s.shift(1)) / 2

    def debt(self) -> pd.Series:
        return (
            self.z("short_term_debt")
            + self.z("current_portion_lt_debt")
            + self.z("long_term_debt")
            + self.z("operating_lease_liabilities")
        )

    def cash(self) -> pd.Series:
        return self.z("cash_and_equivalents") + self.z("short_term_investments")


def div(a: pd.Series, b: pd.Series, positive_denominator: bool = True) -> pd.Series:
    """a / b, NaN when b is missing, zero or (optionally) negative."""
    mask = b > 0 if positive_denominator else b != 0
    return (a / b).where(mask)


def growth(s: pd.Series) -> pd.Series:
    return div(s - s.shift(1), s.shift(1).abs())


def cagr_over(s: pd.Series, years: int) -> float | None:
    if len(s) <= years:
        return None
    start, end = s.iloc[-years - 1], s.iloc[-1]
    if not (np.isfinite(start) and np.isfinite(end)) or start <= 0 or end <= 0:
        return None
    return float((end / start) ** (1 / years) - 1)


def _book(inputs: AnalysisInputs, periods: int = 12) -> Book:
    wide = build_periods(_facts(inputs), "FY")
    if wide.shape[1] < 1:
        raise InsufficientDataError("No fiscal years stored")
    return Book(wide.iloc[:, -periods:])


def _clean(values: pd.Series) -> list[float | None]:
    return [None if v is None or not np.isfinite(v) else float(v) for v in values.astype(float)]


def _table(
    book: Book, rows: list[tuple[str, str, pd.Series, str]], extra: dict[str, Any] | None = None
) -> AnalysisOutput:
    """rows: (code, label, values, unit) where unit ∈ {pct, x, days, amount, per_share, count}."""
    return AnalysisOutput(
        "table",
        {
            "periods": book.periods,
            "rows": [
                {"code": c, "label": lbl, "unit": u, "values": _clean(v)} for c, lbl, v, u in rows
            ],
            **(extra or {}),
        },
    )


class FyParams(BaseModel):
    periods: int = Field(10, ge=2, le=20)


def _register(
    analysis_id: str,
    title: str,
    description: str,
    params: type[BaseModel] = FyParams,
    requires: set[str] | None = None,
    version: str = "1.0.0",
) -> Callable[
    [Callable[[AnalysisInputs, Any], AnalysisOutput]],
    Callable[[AnalysisInputs, Any], AnalysisOutput],
]:
    return analysis(
        id=analysis_id,
        version=version,
        title=title,
        description=description,
        applies_to=COMPANY,
        params=params,
        result_kind="table",
        requires=requires or {"fundamentals"},
    )


def tax_rate(b: Book) -> pd.Series:
    return div(b["income_tax"], b["pretax_income"]).clip(0, 0.35).fillna(0.21)


def invested_capital(b: Book) -> pd.Series:
    return b.debt() + b["total_equity"] - b.cash()


# ── F-010 growth ─────────────────────────────────────────────────────────────


@_register(
    "F-010",
    "Growth",
    "YoY growth of revenue, gross profit, EBITDA, EBIT, net income, diluted EPS, FCF, FCF per "
    "share, book value per share and dividends per share; CAGR over 3/5/10 years, growth "
    "stability (std of YoY) and diluted share-count CAGR.",
)
def growth_metrics(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    shares = b["shares_diluted_wavg"].fillna(b["shares_outstanding"])
    series = {
        "revenue": b["revenue"],
        "gross_profit": b["gross_profit"],
        "ebitda": b["ebitda"],
        "operating_income": b["operating_income"],
        "net_income": b["net_income"],
        "eps_diluted": b["eps_diluted"],
        "fcf": b["fcf"],
        "fcf_per_share": div(b["fcf"], shares),
        "bvps": div(b["total_equity"], shares),
        "dps": div(b["dividends_paid"], shares),
    }
    rows = [(f"{k}_yoy", f"{k.replace('_', ' ')} YoY", growth(v), "pct") for k, v in series.items()]
    summary = {
        k: {
            "cagr_3y": cagr_over(v.dropna(), 3),
            "cagr_5y": cagr_over(v.dropna(), 5),
            "cagr_10y": cagr_over(v.dropna(), 10),
            "yoy_std": _f(growth(v).std()),
        }
        for k, v in series.items()
    }
    return _table(
        Book(b.wide.iloc[:, 1:]),
        [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows],
        {"summary": summary, "share_count_cagr_5y": cagr_over(shares.dropna(), 5)},
    )


def _f(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


# ── F-011 margins ────────────────────────────────────────────────────────────


@_register(
    "F-011",
    "Margins",
    "Gross, EBITDA, EBIT, pre-tax, net, FCF and CFO margins; R&D, SG&A and SBC as % of revenue; "
    "incremental EBIT margin ΔEBIT / Δrevenue.",
)
def margins(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods)
    rev = b["revenue"]
    rows = [
        ("gross_margin", "Gross margin", div(b["gross_profit"], rev), "pct"),
        ("ebitda_margin", "EBITDA margin", div(b["ebitda"], rev), "pct"),
        ("ebit_margin", "EBIT margin", div(b["operating_income"], rev), "pct"),
        ("pretax_margin", "Pre-tax margin", div(b["pretax_income"], rev), "pct"),
        ("net_margin", "Net margin", div(b["net_income"], rev), "pct"),
        ("fcf_margin", "FCF margin", div(b["fcf"], rev), "pct"),
        ("cfo_margin", "CFO margin", div(b["cfo"], rev), "pct"),
        ("rnd_pct", "R&D % revenue", div(b["rnd"], rev), "pct"),
        ("sga_pct", "SG&A % revenue", div(b["sga"], rev), "pct"),
        ("sbc_pct", "SBC % revenue", div(b["sbc"], rev), "pct"),
        (
            "incremental_ebit_margin",
            "Incremental EBIT margin",
            div(b["operating_income"].diff(), rev.diff(), positive_denominator=False),
            "pct",
        ),
    ]
    return _table(b, rows)


# ── F-012 returns on capital / F-013 DuPont ──────────────────────────────────


class ReturnsParams(FyParams):
    wacc: float | None = Field(None, ge=0, le=0.3, description="For the ROIC − WACC spread")


@_register(
    "F-012",
    "Returns on capital",
    "ROE = NI / avg equity, ROA = NI / avg assets, ROIC = EBIT(1 − τ) / avg(debt + leases + "
    "equity − cash) with τ clipped to [0, 35 %], ROCE = EBIT / (assets − current liabilities), "
    "CROIC = FCF / avg invested capital, ROTE = NI / avg tangible equity; ROIC − WACC.",
    ReturnsParams,
)
def returns_on_capital(inputs: AnalysisInputs, params: ReturnsParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    ic = invested_capital(b)
    avg_ic = (ic + ic.shift(1)) / 2
    tangible = b["total_equity"] - b.z("goodwill") - b.z("intangibles")
    nopat = b["operating_income"] * (1 - tax_rate(b))
    roic = div(nopat, avg_ic)
    rows = [
        ("roe", "ROE", div(b["net_income"], b.avg("total_equity")), "pct"),
        ("roa", "ROA", div(b["net_income"], b.avg("total_assets")), "pct"),
        ("roic", "ROIC", roic, "pct"),
        (
            "roce",
            "ROCE",
            div(b["operating_income"], b["total_assets"] - b["total_current_liabilities"]),
            "pct",
        ),
        ("croic", "CROIC", div(b["fcf"], avg_ic), "pct"),
        ("rote", "ROTE", div(b["net_income"], (tangible + tangible.shift(1)) / 2), "pct"),
        ("tax_rate", "Effective tax rate (clipped)", tax_rate(b), "pct"),
    ]
    if params.wacc is not None:
        rows.append(("roic_minus_wacc", "ROIC − WACC", roic - params.wacc, "pct"))
    return _table(Book(b.wide.iloc[:, 1:]), [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows])


@_register(
    "F-013",
    "DuPont",
    "3-step ROE = net margin × asset turnover × equity multiplier; 5-step = tax burden (NI/EBT) × "
    "interest burden (EBT/EBIT) × EBIT margin × asset turnover × equity multiplier (average "
    "balances); the product is checked against ROE.",
)
def dupont(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    turnover = div(b["revenue"], b.avg("total_assets"))
    multiplier = div(b.avg("total_assets"), b.avg("total_equity"))
    rows = [
        ("net_margin", "Net margin", div(b["net_income"], b["revenue"]), "pct"),
        ("asset_turnover", "Asset turnover", turnover, "x"),
        ("equity_multiplier", "Equity multiplier", multiplier, "x"),
        ("tax_burden", "Tax burden (NI/EBT)", div(b["net_income"], b["pretax_income"], False), "x"),
        (
            "interest_burden",
            "Interest burden (EBT/EBIT)",
            div(b["pretax_income"], b["operating_income"], False),
            "x",
        ),
        ("ebit_margin", "EBIT margin", div(b["operating_income"], b["revenue"]), "pct"),
        (
            "roe_dupont",
            "ROE (product)",
            div(b["net_income"], b["revenue"]) * turnover * multiplier,
            "pct",
        ),
    ]
    return _table(Book(b.wide.iloc[:, 1:]), [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows])


# ── F-014 liquidity / F-015 solvency / F-016 efficiency ──────────────────────


@_register(
    "F-014",
    "Liquidity",
    "Current ratio, quick ratio (cash + STI + receivables) / current liabilities, cash ratio, "
    "defensive interval in days (liquid assets / daily cash opex), operating cash-flow ratio.",
)
def liquidity(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods)
    cl = b["total_current_liabilities"]
    liquid = b.cash() + b.z("receivables")
    daily_opex = (b["revenue"] - b["operating_income"] - b.z("d_and_a")) / DAYS
    rows = [
        ("current_ratio", "Current ratio", div(b["total_current_assets"], cl), "x"),
        ("quick_ratio", "Quick ratio", div(liquid, cl), "x"),
        ("cash_ratio", "Cash ratio", div(b.cash(), cl), "x"),
        ("defensive_interval", "Defensive interval", div(liquid, daily_opex), "days"),
        ("ocf_ratio", "Operating cash-flow ratio", div(b["cfo"], cl), "x"),
    ]
    return _table(b, rows)


@_register(
    "F-015",
    "Solvency & leverage",
    "Total debt (incl. operating leases, D-FU-3), net debt, D/E, D/capital, D/assets, ND/EBITDA, "
    "ND/FCF, EBIT and EBITDA interest coverage, FCCR = (EBITDA − capex) / interest, equity "
    "multiplier.",
)
def solvency(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods)
    debt, net_debt, equity = b.debt(), b.debt() - b.cash(), b["total_equity"]
    interest = b["interest_expense"]
    rows = [
        ("total_debt", "Total debt (incl. leases)", debt, "amount"),
        ("net_debt", "Net debt", net_debt, "amount"),
        ("debt_to_equity", "Debt / equity", div(debt, equity), "x"),
        ("debt_to_capital", "Debt / capital", div(debt, debt + equity), "pct"),
        ("debt_to_assets", "Debt / assets", div(debt, b["total_assets"]), "pct"),
        ("nd_ebitda", "Net debt / EBITDA", div(net_debt, b["ebitda"]), "x"),
        ("nd_fcf", "Net debt / FCF", div(net_debt, b["fcf"]), "x"),
        ("ebit_coverage", "EBIT / interest", div(b["operating_income"], interest), "x"),
        ("ebitda_coverage", "EBITDA / interest", div(b["ebitda"], interest), "x"),
        ("fccr", "Fixed-charge coverage", div(b["ebitda"] - b.z("capex"), interest), "x"),
        ("equity_multiplier", "Equity multiplier", div(b["total_assets"], equity), "x"),
    ]
    return _table(b, rows)


@_register(
    "F-016",
    "Efficiency",
    "Asset and fixed-asset turnover; DSO = avg receivables / revenue × 365, DIO = avg inventory "
    "/ COGS × 365, DPO = avg payables / COGS × 365, cash conversion cycle; working capital "
    "(receivables + inventory − payables) % revenue.",
)
def efficiency(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    cogs = b["cost_of_revenue"]
    dso = div(b.avg("receivables"), b["revenue"]) * DAYS
    dio = div(b.avg("inventory"), cogs) * DAYS
    dpo = div(b.avg("accounts_payable"), cogs) * DAYS
    wc = b.z("receivables") + b.z("inventory") - b.z("accounts_payable")
    rows = [
        ("asset_turnover", "Asset turnover", div(b["revenue"], b.avg("total_assets")), "x"),
        ("fixed_asset_turnover", "Fixed-asset turnover", div(b["revenue"], b.avg("ppe_net")), "x"),
        ("dso", "DSO", dso, "days"),
        ("dio", "DIO", dio, "days"),
        ("dpo", "DPO", dpo, "days"),
        ("ccc", "Cash conversion cycle", dso.fillna(0) + dio.fillna(0) - dpo.fillna(0), "days"),
        ("wc_pct_revenue", "Working capital % revenue", div(wc, b["revenue"]), "pct"),
    ]
    return _table(Book(b.wide.iloc[:, 1:]), [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows])


# ── F-017 cash generation / F-018 capital allocation / F-019 per share ───────


@_register(
    "F-017",
    "Cash generation & quality",
    "CFO, capex, FCF, FCF − SBC; FCF/NI, CFO/NI, capex/revenue, capex/D&A; Sloan accruals "
    "(NI − CFO) / avg assets.",
)
def cash_quality(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    rows = [
        ("cfo", "CFO", b["cfo"], "amount"),
        ("capex", "Capex", b["capex"], "amount"),
        ("fcf", "FCF", b["fcf"], "amount"),
        ("fcf_minus_sbc", "FCF − SBC", b["fcf_minus_sbc"], "amount"),
        ("fcf_to_ni", "FCF / NI", div(b["fcf"], b["net_income"]), "x"),
        ("cfo_to_ni", "CFO / NI", div(b["cfo"], b["net_income"]), "x"),
        ("capex_to_revenue", "Capex / revenue", div(b["capex"], b["revenue"]), "pct"),
        ("capex_to_da", "Capex / D&A", div(b["capex"], b["d_and_a"]), "x"),
        (
            "sloan_accruals",
            "Sloan accruals",
            div(b["net_income"] - b["cfo"], b.avg("total_assets")),
            "pct",
        ),
    ]
    return _table(Book(b.wide.iloc[:, 1:]), [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows])


def _market_cap(inputs: AnalysisInputs, b: Book, facts: pd.DataFrame) -> pd.Series:
    """Close at each FY end × shares outstanding (NaN when prices are not stored)."""
    prices = inputs.data.get("prices")
    out = pd.Series(np.nan, index=b.periods, dtype=float)
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return out
    ends = facts[facts["period_type"] == "FY"].drop_duplicates("fiscal_year", keep="last")
    end_of = {
        f"FY{int(y)}": pd.Timestamp(e)
        for y, e in zip(ends["fiscal_year"], ends["period_end"], strict=True)
    }
    close = prices["close"]
    for period in b.periods:
        end = end_of.get(period)
        if end is None:
            continue
        before = close.loc[close.index <= end]
        if len(before) and (close.index[0] <= end):
            out[period] = float(before.iloc[-1])
    return out * b["shares_outstanding"]


@_register(
    "F-018",
    "Capital allocation",
    "Dividends, payout on NI and on FCF, DPS and its growth, consecutive years of DPS increase, "
    "gross and net buybacks, shareholder yield ((dividends + net buybacks) / market cap), net "
    "debt issuance, M&A spend, reinvestment rate (capex − D&A + ΔWC) / NOPAT, sustainable growth "
    "b·ROE.",
    requires={"fundamentals", "prices"},
)
def capital_allocation(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    facts = _facts(inputs)
    b = _book(inputs, params.periods + 1)
    shares = b["shares_diluted_wavg"].fillna(b["shares_outstanding"])
    dps = div(b.z("dividends_paid"), shares)
    net_buybacks = b.z("buybacks") - b.z("share_issuance")
    wc = b.z("receivables") + b.z("inventory") - b.z("accounts_payable")
    nopat = b["operating_income"] * (1 - tax_rate(b))
    roe = div(b["net_income"], b.avg("total_equity"))
    retention = 1 - div(b.z("dividends_paid"), b["net_income"])
    mcap = _market_cap(inputs, b, facts)
    increases = (dps.diff() > 0).astype(int)
    streak = 0
    for flag in reversed(increases.iloc[1:].tolist()):
        if not flag:
            break
        streak += 1
    rows = [
        ("dividends", "Dividends paid", b.z("dividends_paid"), "amount"),
        ("payout_ni", "Payout (NI)", div(b.z("dividends_paid"), b["net_income"]), "pct"),
        ("payout_fcf", "Payout (FCF)", div(b.z("dividends_paid"), b["fcf"]), "pct"),
        ("dps", "DPS", dps, "per_share"),
        ("dps_growth", "DPS growth", growth(dps), "pct"),
        ("buybacks_gross", "Buybacks (gross)", b.z("buybacks"), "amount"),
        ("buybacks_net", "Buybacks (net of issuance)", net_buybacks, "amount"),
        (
            "shareholder_yield",
            "Shareholder yield",
            div(b.z("dividends_paid") + net_buybacks, mcap),
            "pct",
        ),
        (
            "net_debt_issuance",
            "Net debt issuance",
            b.z("debt_issued") - b.z("debt_repaid"),
            "amount",
        ),
        ("acquisitions", "M&A spend", b.z("acquisitions"), "amount"),
        (
            "reinvestment_rate",
            "Reinvestment rate",
            div(b.z("capex") - b.z("d_and_a") + wc.diff(), nopat),
            "pct",
        ),
        ("sustainable_growth", "Sustainable growth b·ROE", retention * roe, "pct"),
    ]
    return _table(
        Book(b.wide.iloc[:, 1:]),
        [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows],
        {"dps_increase_streak": streak},
    )


@_register(
    "F-019",
    "Per-share",
    "EPS basic/diluted, BVPS, tangible BVPS, FCF/share, sales/share, DPS, cash/share on diluted "
    "weighted shares (period-end shares for book values).",
)
def per_share(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods)
    wavg = b["shares_diluted_wavg"].fillna(b["shares_outstanding"])
    end = b["shares_outstanding"].fillna(wavg)
    tangible = b["total_equity"] - b.z("goodwill") - b.z("intangibles")
    rows = [
        ("eps_basic", "EPS basic", b["eps_basic"], "per_share"),
        ("eps_diluted", "EPS diluted", b["eps_diluted"], "per_share"),
        ("bvps", "Book value / share", div(b["total_equity"], end, False), "per_share"),
        ("tbvps", "Tangible BV / share", div(tangible, end, False), "per_share"),
        ("fcf_per_share", "FCF / share", div(b["fcf"], wavg), "per_share"),
        ("sales_per_share", "Sales / share", div(b["revenue"], wavg), "per_share"),
        ("dps", "DPS", div(b.z("dividends_paid"), wavg), "per_share"),
        ("cash_per_share", "Cash / share", div(b.cash(), end), "per_share"),
    ]
    return _table(b, rows)


# ── F-023 operating leverage / F-024 productivity / F-025 sector KPIs ────────


@_register(
    "F-023",
    "Operating leverage",
    "DOL = %ΔEBIT / %Δrevenue, DFL = %ΔEPS / %ΔEBIT, DTL = DOL × DFL; fixed-cost proxy = "
    "1 − Δ(costs)/Δ(revenue) where costs = revenue − EBIT.",
)
def operating_leverage(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    dol = div(growth(b["operating_income"]), growth(b["revenue"]), False)
    dfl = div(growth(b["eps_diluted"]), growth(b["operating_income"]), False)
    costs = b["revenue"] - b["operating_income"]
    rows = [
        ("dol", "DOL", dol, "x"),
        ("dfl", "DFL", dfl, "x"),
        ("dtl", "DTL", dol * dfl, "x"),
        (
            "fixed_cost_proxy",
            "Fixed-cost proxy",
            1 - div(costs.diff(), b["revenue"].diff(), False),
            "pct",
        ),
    ]
    return _table(Book(b.wide.iloc[:, 1:]), [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows])


@_register(
    "F-024",
    "Productivity",
    "Revenue, net income and FCF per employee (employees as filed on the cover page).",
)
def productivity(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods)
    if b["employees"].isna().all():
        raise InsufficientDataError("The filer does not report its number of employees in XBRL")
    rows = [
        ("employees", "Employees", b["employees"], "count"),
        ("revenue_per_employee", "Revenue / employee", div(b["revenue"], b["employees"]), "amount"),
        (
            "ni_per_employee",
            "Net income / employee",
            div(b["net_income"], b["employees"]),
            "amount",
        ),
        ("fcf_per_employee", "FCF / employee", div(b["fcf"], b["employees"]), "amount"),
    ]
    return _table(b, rows)


def sector_set(conventions: dict[str, Any] | Any) -> str:
    """Threshold/KPI set from the type & industry (D-FU-5): general, banks, insurers, reits,
    utilities, software."""
    sub = str(conventions.get("asset_subclass") or "")
    industry = str(conventions.get("industry") or "")
    sector = str(conventions.get("sector") or "")
    if sub == "reit" or industry.startswith("REIT"):
        return "reits"
    if "Bank" in industry:
        return "banks"
    if "Insurance" in industry:
        return "insurers"
    if sector == "Utilities":
        return "utilities"
    if industry.startswith("Software") or "Internet" in industry:
        return "software"
    return "general"


@_register(
    "F-025",
    "Sector-specific KPIs",
    "Software: Rule of 40 (revenue growth + FCF margin); retail & industrial: inventory turns; "
    "banks: net interest income / assets (NIM proxy) and cost / income; REITs: FFO ≈ NI + D&A "
    "and FFO payout (approximation without gains on sale).",
)
def sector_kpis(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    kind = sector_set(inputs.conventions)
    rows: list[tuple[str, str, pd.Series, str]] = [
        ("rule_of_40", "Rule of 40", growth(b["revenue"]) + div(b["fcf"], b["revenue"]), "pct"),
        ("inventory_turns", "Inventory turns", div(b["cost_of_revenue"], b.avg("inventory")), "x"),
    ]
    if kind == "banks":
        nii = b["interest_income"] - b.z("interest_expense")
        rows += [
            ("nim_proxy", "Net interest income / assets", div(nii, b.avg("total_assets")), "pct"),
            (
                "cost_income",
                "Cost / income",
                div(b["revenue"] - b["pretax_income"], b["revenue"]),
                "pct",
            ),
        ]
    if kind == "reits":
        ffo = b["net_income"] + b.z("d_and_a")
        rows += [
            ("ffo", "FFO (≈ NI + D&A)", ffo, "amount"),
            ("ffo_payout", "FFO payout", div(b.z("dividends_paid"), ffo), "pct"),
        ]
    return _table(
        Book(b.wide.iloc[:, 1:]),
        [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows],
        {"sector_set": kind},
    )


# ── S-001 health scorecard ───────────────────────────────────────────────────

# (metric, pillar, bad, good): linear score 0 at `bad`, 100 at `good` (either direction).
THRESHOLDS: dict[str, list[tuple[str, str, float, float]]] = {
    "general": [
        ("revenue_cagr_3y", "growth", -0.05, 0.15),
        ("eps_cagr_3y", "growth", -0.05, 0.15),
        ("fcf_cagr_3y", "growth", -0.10, 0.15),
        ("ebit_margin", "profitability", 0.0, 0.25),
        ("roic", "profitability", 0.0, 0.20),
        ("roe", "profitability", 0.0, 0.20),
        ("nd_ebitda", "balance_sheet", 4.0, 0.0),
        ("ebit_coverage", "balance_sheet", 1.5, 15.0),
        ("current_ratio", "balance_sheet", 0.8, 2.0),
        ("fcf_margin", "cash_flow", -0.05, 0.20),
        ("fcf_to_ni", "cash_flow", 0.3, 1.2),
        ("cfo_to_ni", "cash_flow", 0.5, 1.3),
    ],
    "software": [
        ("revenue_cagr_3y", "growth", 0.0, 0.30),
        ("eps_cagr_3y", "growth", -0.05, 0.25),
        ("fcf_cagr_3y", "growth", -0.10, 0.25),
        ("ebit_margin", "profitability", -0.05, 0.30),
        ("roic", "profitability", 0.0, 0.25),
        ("roe", "profitability", 0.0, 0.25),
        ("nd_ebitda", "balance_sheet", 3.0, -1.0),
        ("ebit_coverage", "balance_sheet", 2.0, 20.0),
        ("current_ratio", "balance_sheet", 0.8, 2.5),
        ("fcf_margin", "cash_flow", 0.0, 0.30),
        ("fcf_to_ni", "cash_flow", 0.5, 1.5),
        ("cfo_to_ni", "cash_flow", 0.7, 1.5),
    ],
    "utilities": [
        ("revenue_cagr_3y", "growth", -0.03, 0.06),
        ("eps_cagr_3y", "growth", -0.03, 0.07),
        ("fcf_cagr_3y", "growth", -0.10, 0.10),
        ("ebit_margin", "profitability", 0.05, 0.25),
        ("roic", "profitability", 0.02, 0.08),
        ("roe", "profitability", 0.04, 0.12),
        ("nd_ebitda", "balance_sheet", 7.0, 3.0),
        ("ebit_coverage", "balance_sheet", 1.5, 5.0),
        ("current_ratio", "balance_sheet", 0.5, 1.2),
        ("fcf_margin", "cash_flow", -0.20, 0.10),
        ("fcf_to_ni", "cash_flow", -0.5, 1.0),
        ("cfo_to_ni", "cash_flow", 1.0, 2.5),
    ],
    "banks": [
        ("revenue_cagr_3y", "growth", -0.05, 0.10),
        ("eps_cagr_3y", "growth", -0.05, 0.12),
        ("roe", "profitability", 0.04, 0.15),
        ("roa", "profitability", 0.002, 0.015),
    ],
    "insurers": [
        ("revenue_cagr_3y", "growth", -0.05, 0.10),
        ("eps_cagr_3y", "growth", -0.05, 0.12),
        ("roe", "profitability", 0.04, 0.15),
        ("roa", "profitability", 0.002, 0.03),
    ],
    "reits": [
        ("revenue_cagr_3y", "growth", -0.03, 0.08),
        ("fcf_cagr_3y", "growth", -0.10, 0.10),
        ("ebit_margin", "profitability", 0.1, 0.45),
        ("roe", "profitability", 0.0, 0.10),
        ("nd_ebitda", "balance_sheet", 9.0, 4.0),
        ("ebit_coverage", "balance_sheet", 1.5, 5.0),
        ("cfo_to_ni", "cash_flow", 0.8, 2.0),
    ],
}


def metric_history(b: Book) -> dict[str, pd.Series]:
    """Per-year values of every scorecard metric (inputs to absolute and percentile scores)."""
    ic = invested_capital(b)
    nopat = b["operating_income"] * (1 - tax_rate(b))
    net_debt = b.debt() - b.cash()

    def rolling_cagr(s: pd.Series, years: int = 3) -> pd.Series:
        ratio = s / s.shift(years)
        return (ratio ** (1 / years) - 1).where((s > 0) & (s.shift(years) > 0))

    return {
        "revenue_cagr_3y": rolling_cagr(b["revenue"]),
        "eps_cagr_3y": rolling_cagr(b["eps_diluted"]),
        "fcf_cagr_3y": rolling_cagr(b["fcf"]),
        "ebit_margin": div(b["operating_income"], b["revenue"]),
        "roic": div(nopat, (ic + ic.shift(1)) / 2),
        "roe": div(b["net_income"], b.avg("total_equity")),
        "roa": div(b["net_income"], b.avg("total_assets")),
        "nd_ebitda": div(net_debt, b["ebitda"]),
        "ebit_coverage": div(b["operating_income"], b["interest_expense"]),
        "current_ratio": div(b["total_current_assets"], b["total_current_liabilities"]),
        "fcf_margin": div(b["fcf"], b["revenue"]),
        "fcf_to_ni": div(b["fcf"], b["net_income"]),
        "cfo_to_ni": div(b["cfo"], b["net_income"]),
    }


# Valuation thresholds (bad, good): lower multiples / higher yields score higher.
VALUATION_THRESHOLDS = (("pe", 35.0, 12.0), ("ev_ebitda", 25.0, 8.0), ("fcf_yield", 0.01, 0.07))


def _valuation_pillar(
    inputs: AnalysisInputs,
    absolute_weight: float,
    drill: list[dict[str, Any]],
    pillars: dict[str, list[float]],
) -> str:
    """Scores current multiples against thresholds and their own point-in-time history
    (V-002 weekly, 10 years); appended to the drill-down. Returns a status note."""
    if not isinstance(inputs.data.get("prices"), pd.DataFrame):
        return "needs stored prices"
    from fda_core.analyses.valuation import HistoryParams, pit_multiples

    try:
        history = pit_multiples(inputs, HistoryParams(years=10)).payload
    except (InsufficientDataError, NotApplicableError) as exc:
        return str(exc)
    for metric, bad, good in VALUATION_THRESHOLDS:
        series = pd.Series(history["lines"].get(metric) or [], dtype=float).dropna()
        if series.empty:
            continue
        value = float(series.iloc[-1])
        absolute = linear_score(value, bad, good)
        rank = (series <= value).mean() if good > bad else (series >= value).mean()
        percentile = float(rank * 100) if len(series) >= 12 else absolute
        score = absolute_weight * absolute + (1 - absolute_weight) * percentile
        drill.append(
            {
                "metric": metric,
                "pillar": "valuation",
                "value": value,
                "bad": bad,
                "good": good,
                "absolute": absolute,
                "history_percentile": percentile,
                "score": score,
                "years": history.get("years"),
            }
        )
        pillars.setdefault("valuation", []).append(score)
    return "scored" if "valuation" in pillars else "multiples unavailable"


def linear_score(value: float, bad: float, good: float) -> float:
    return float(np.clip((value - bad) / (good - bad), 0, 1) * 100)


class ScorecardParams(FyParams):
    absolute_weight: float = Field(
        0.7, ge=0, le=1, description="D-FU-5: 70 % absolute / 30 % history"
    )
    threshold_set: Literal[
        "auto", "general", "software", "utilities", "banks", "insurers", "reits"
    ] = "auto"


@_register(
    "S-001",
    "Health scorecard",
    "Pillars Growth, Profitability, Balance sheet, Cash flow (0–100): each metric scores "
    "linearly between sector-calibrated bad/good thresholds (70 %) blended with its percentile "
    "in the company's own history (30 %); pillar = mean of available metrics; overall = mean of "
    "pillars. Valuation pillar (when prices are stored): TTM P/E, EV/EBITDA and FCF yield vs "
    "thresholds and vs their point-in-time history (V-001/V-002). Full drill-down.",
    ScorecardParams,
    {"fundamentals", "fundamentals_all", "prices", "actions"},
    "1.1.0",
)
def health_scorecard(inputs: AnalysisInputs, params: ScorecardParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 4)
    kind = (
        sector_set(inputs.conventions) if params.threshold_set == "auto" else params.threshold_set
    )
    history = metric_history(b)
    drill: list[dict[str, Any]] = []
    pillars: dict[str, list[float]] = {}
    for metric, pillar, bad, good in THRESHOLDS[kind]:
        series = history[metric].dropna()
        if series.empty:
            drill.append(
                {
                    "metric": metric,
                    "pillar": pillar,
                    "value": None,
                    "score": None,
                    "reason": "input missing",
                }
            )
            continue
        value = float(series.iloc[-1])
        absolute = linear_score(value, bad, good)
        better_high = good > bad
        rank = (series <= value).mean() if better_high else (series >= value).mean()
        percentile = float(rank * 100) if len(series) >= 3 else absolute
        score = params.absolute_weight * absolute + (1 - params.absolute_weight) * percentile
        drill.append(
            {
                "metric": metric,
                "pillar": pillar,
                "value": value,
                "bad": bad,
                "good": good,
                "absolute": absolute,
                "history_percentile": percentile,
                "score": score,
                "years": len(series),
            }
        )
        pillars.setdefault(pillar, []).append(score)
    valuation_note = _valuation_pillar(inputs, params.absolute_weight, drill, pillars)
    pillar_scores = {p: float(np.mean(v)) for p, v in pillars.items()}
    overall = float(np.mean(list(pillar_scores.values()))) if pillar_scores else None
    return AnalysisOutput(
        "table",
        {
            "overall": overall,
            "pillars": pillar_scores,
            "valuation": valuation_note,
            "threshold_set": kind,
            "period": b.periods[-1],
            "drilldown": drill,
        },
    )


# ── S-002 Piotroski / S-003 Altman / S-009 Ohlson ────────────────────────────


@_register(
    "S-002",
    "Piotroski F-score",
    "Nine binary tests on the last two fiscal years: ROA > 0, CFO > 0, ΔROA > 0, CFO > NI, "
    "Δleverage (LT debt / assets) < 0, Δcurrent ratio > 0, no dilution, Δgross margin > 0, "
    "Δasset turnover > 0.",
)
def piotroski(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, 4)
    if len(b.periods) < 3:
        raise InsufficientDataError("Piotroski needs three fiscal years (two with averages)")
    roa = div(b["net_income"], b["total_assets"].shift(1))
    cfo_a = div(b["cfo"], b["total_assets"].shift(1))
    leverage = div(b.z("long_term_debt"), b.avg("total_assets"))
    current = div(b["total_current_assets"], b["total_current_liabilities"])
    shares = b["shares_outstanding"].fillna(b["shares_diluted_wavg"])
    gm = div(b["gross_profit"], b["revenue"])
    turnover = div(b["revenue"], b["total_assets"].shift(1))
    t, p = -1, -2

    def flag(value: Any) -> bool | None:
        return (
            None
            if value is None or (isinstance(value, float) and not np.isfinite(value))
            else bool(value)
        )

    tests = {
        "roa_positive": flag(roa.iloc[t] > 0 if np.isfinite(roa.iloc[t]) else np.nan),
        "cfo_positive": flag(b["cfo"].iloc[t] > 0 if np.isfinite(b["cfo"].iloc[t]) else np.nan),
        "roa_improving": flag(
            roa.iloc[t] > roa.iloc[p] if np.isfinite(roa.iloc[t] - roa.iloc[p]) else np.nan
        ),
        "accruals": flag(
            cfo_a.iloc[t] > roa.iloc[t] if np.isfinite(cfo_a.iloc[t] - roa.iloc[t]) else np.nan
        ),
        "leverage_down": flag(
            leverage.iloc[t] < leverage.iloc[p]
            if np.isfinite(leverage.iloc[t] - leverage.iloc[p])
            else np.nan
        ),
        "liquidity_up": flag(
            current.iloc[t] > current.iloc[p]
            if np.isfinite(current.iloc[t] - current.iloc[p])
            else np.nan
        ),
        "no_dilution": flag(
            shares.iloc[t] <= shares.iloc[p] * 1.001
            if np.isfinite(shares.iloc[t] - shares.iloc[p])
            else np.nan
        ),
        "gross_margin_up": flag(
            gm.iloc[t] > gm.iloc[p] if np.isfinite(gm.iloc[t] - gm.iloc[p]) else np.nan
        ),
        "turnover_up": flag(
            turnover.iloc[t] > turnover.iloc[p]
            if np.isfinite(turnover.iloc[t] - turnover.iloc[p])
            else np.nan
        ),
    }
    available = [v for v in tests.values() if v is not None]
    return AnalysisOutput(
        "table",
        {
            "score": sum(available),
            "tests_available": len(available),
            "tests": tests,
            "period": b.periods[-1],
            "reading": "strong (8–9)"
            if sum(available) >= 8
            else "weak (0–2)"
            if sum(available) <= 2
            else "neutral",
        },
    )


MANUFACTURING_SECTORS = {"Industrials", "Basic Materials", "Consumer Cyclical", "Energy"}


class AltmanParams(FyParams):
    variant: Literal["auto", "z", "z_double_prime"] = "auto"


@_register(
    "S-003",
    "Altman Z / Z″",
    "Auto variant (D-FU-6): original Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5 for US "
    "manufacturers (X4 = market cap / total liabilities), Z″ = 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4 "
    "for others (X4 = book equity / liabilities); zones; hidden for financials.",
    AltmanParams,
    requires={"fundamentals", "prices"},
)
def altman(inputs: AnalysisInputs, params: AltmanParams) -> AnalysisOutput:
    if (
        sector_set(inputs.conventions) in {"banks", "insurers"}
        or inputs.conventions.get("sector") == "Financial Services"
    ):
        raise InsufficientDataError("Altman Z is not meaningful for financial companies")
    facts = _facts(inputs)
    b = _book(inputs, params.periods)
    ta, tl = b["total_assets"], b["total_liabilities"]
    x1 = div(b["total_current_assets"] - b["total_current_liabilities"], ta)
    x2 = div(b["retained_earnings"], ta)
    x3 = div(b["operating_income"], ta)
    x5 = div(b["revenue"], ta)
    manufacturer = inputs.conventions.get(
        "sector"
    ) in MANUFACTURING_SECTORS and inputs.conventions.get("country") in {None, "US"}
    variant = (
        params.variant if params.variant != "auto" else ("z" if manufacturer else "z_double_prime")
    )
    if variant == "z":
        x4 = div(_market_cap(inputs, b, facts), tl)
        score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        zones = (1.81, 2.99)
    else:
        x4 = div(b["total_equity"], tl)
        score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        zones = (1.1, 2.6)
    last = _f(score.iloc[-1])
    zone = (
        None
        if last is None
        else "distress"
        if last < zones[0]
        else "safe"
        if last > zones[1]
        else "grey"
    )
    return AnalysisOutput(
        "table",
        {
            "variant": variant,
            "periods": b.periods,
            "score": _clean(score),
            "latest": last,
            "zone": zone,
            "zones": {"distress_below": zones[0], "safe_above": zones[1]},
            "components": {
                "x1": _clean(x1),
                "x2": _clean(x2),
                "x3": _clean(x3),
                "x4": _clean(x4),
                "x5": _clean(x5),
            },
        },
    )


@_register(
    "S-009",
    "Ohlson O-score",
    "Ohlson (1980) logit: O = −1.32 − 0.407·SIZE + 6.03·TLTA − 1.43·WCTA + 0.0757·CLCA − "
    "2.37·NITA − 1.83·FUTL + 0.285·INTWO − 1.72·OENEG − 0.521·CHIN; P(default) = 1/(1 + e^−O). "
    "SIZE = log(total assets in millions) without the GNP deflator (approximation, flagged).",
)
def ohlson(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    ta, tl, ni = b["total_assets"], b["total_liabilities"], b["net_income"]
    size = np.log(ta / 1e6)
    o = (
        -1.32
        - 0.407 * size
        + 6.03 * div(tl, ta)
        - 1.43 * div(b["total_current_assets"] - b["total_current_liabilities"], ta)
        + 0.0757 * div(b["total_current_liabilities"], b["total_current_assets"])
        - 2.37 * div(ni, ta)
        - 1.83 * div(b["cfo"], tl)
        + 0.285 * ((ni < 0) & (ni.shift(1) < 0)).astype(float)
        - 1.72 * (tl > ta).astype(float)
        - 0.521 * div(ni - ni.shift(1), ni.abs() + ni.shift(1).abs())
    )
    probability = 1 / (1 + np.exp(-o))
    return AnalysisOutput(
        "table",
        {
            "periods": b.periods[1:],
            "o_score": _clean(o.iloc[1:]),
            "probability": _clean(probability.iloc[1:]),
            "latest_probability": _f(probability.iloc[-1]),
        },
        (
            "SIZE uses total assets without the GNP price-level deflator: "
            "compare companies relatively.",
        ),
    )


# ── S-004 Beneish / S-006 Montier / S-007 earnings quality ───────────────────


@_register(
    "S-004",
    "Beneish M-score",
    "M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI − 0.172·SGAI + "
    "4.679·TATA − 0.327·LVGI on the last two fiscal years; flag above −1.78 (−2.22 conservative).",
)
def beneish(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    rev, rec, ta = b["revenue"], b["receivables"], b["total_assets"]
    gm = div(rev - b["cost_of_revenue"], rev)
    hard = b["total_current_assets"] + b["ppe_net"]
    dep_rate = div(b["d_and_a"], b["d_and_a"] + b["ppe_net"])
    lev = div(b["total_liabilities"], ta)
    variables = {
        "dsri": div(div(rec, rev), div(rec, rev).shift(1)),
        "gmi": div(gm.shift(1), gm),
        "aqi": div(1 - div(hard, ta), (1 - div(hard, ta)).shift(1)),
        "sgi": div(rev, rev.shift(1)),
        "depi": div(dep_rate.shift(1), dep_rate),
        "sgai": div(div(b["sga"], rev), div(b["sga"], rev).shift(1)),
        "tata": div(b["net_income"] - b["cfo"], ta, False),
        "lvgi": div(lev, lev.shift(1)),
    }
    weights = {
        "dsri": 0.920,
        "gmi": 0.528,
        "aqi": 0.404,
        "sgi": 0.892,
        "depi": 0.115,
        "sgai": -0.172,
        "tata": 4.679,
        "lvgi": -0.327,
    }
    filled = {k: v.fillna(1.0 if k != "tata" else 0.0) for k, v in variables.items()}
    m = -4.84 + sum(weights[k] * filled[k] for k in weights)
    imputed = [k for k, v in variables.items() if not np.isfinite(v.iloc[-1])]
    last = _f(m.iloc[-1])
    return AnalysisOutput(
        "table",
        {
            "periods": b.periods[1:],
            "m_score": _clean(m.iloc[1:]),
            "latest": last,
            "flag": None
            if last is None
            else "likely manipulator"
            if last > -1.78
            else "watch"
            if last > -2.22
            else "unlikely",
            "variables": {k: _f(v.iloc[-1]) for k, v in variables.items()},
            "imputed_neutral": imputed,
        },
        (f"Missing inputs set to neutral (index 1, TATA 0): {', '.join(imputed)}",)
        if imputed
        else (),
    )


@_register(
    "S-006",
    "Montier C-score",
    "Six binary signals (one point each): NI − CFO gap widening, DSO up, DIO up, other current "
    "assets / revenue up, depreciation / PP&E down, total assets growth > 10 %.",
)
def montier(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, 4)
    if len(b.periods) < 3:
        raise InsufficientDataError("Montier C-score needs three fiscal years")
    gap = b["net_income"] - b["cfo"]
    dso = div(b["receivables"], b["revenue"])
    dio = div(b["inventory"], b["cost_of_revenue"])
    oca = div(b["other_current_assets"], b["revenue"])
    dep = div(b["d_and_a"], b["ppe_net"])

    def up(s: pd.Series) -> bool | None:
        return None if not np.isfinite(s.iloc[-1] - s.iloc[-2]) else bool(s.iloc[-1] > s.iloc[-2])

    signals = {
        "ni_cfo_gap_widening": up(gap),
        "dso_up": up(dso),
        "dio_up": up(dio),
        "other_current_assets_up": up(oca),
        "depreciation_rate_down": None if up(dep) is None else not up(dep),
        "asset_growth_over_10pct": None
        if not np.isfinite(growth(b["total_assets"]).iloc[-1])
        else bool(growth(b["total_assets"]).iloc[-1] > 0.10),
    }
    available = [v for v in signals.values() if v is not None]
    return AnalysisOutput(
        "table",
        {
            "score": sum(available),
            "signals_available": len(available),
            "signals": signals,
            "period": b.periods[-1],
            "reading": "high risk (≥ 4)" if sum(available) >= 4 else "low/moderate",
        },
    )


@_register(
    "S-007",
    "Earnings-quality dashboard",
    "Balance-sheet accruals (ΔNOA / avg NOA) and cash-flow accruals ((NI − CFO) / avg assets), "
    "CFO/NI trend, DSO growth vs revenue growth, inventory growth vs sales growth, capitalised "
    "costs (capex / (capex + opex)), one-off frequency (|other non-operating| > 10 % of pre-tax).",
)
def earnings_quality(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, params.periods + 1)
    noa = (b["total_assets"] - b.cash()) - (b["total_liabilities"] - b.debt())
    opex = b["revenue"] - b["operating_income"]
    rows = [
        ("bs_accruals", "Balance-sheet accruals", div(noa.diff(), (noa + noa.shift(1)) / 2), "pct"),
        (
            "cf_accruals",
            "Cash-flow accruals",
            div(b["net_income"] - b["cfo"], b.avg("total_assets")),
            "pct",
        ),
        ("cfo_to_ni", "CFO / NI", div(b["cfo"], b["net_income"]), "x"),
        (
            "dso_growth_minus_revenue_growth",
            "DSO growth − revenue growth",
            growth(div(b["receivables"], b["revenue"])) - growth(b["revenue"]),
            "pct",
        ),
        (
            "inventory_minus_sales_growth",
            "Inventory growth − sales growth",
            growth(b["inventory"]) - growth(b["revenue"]),
            "pct",
        ),
        ("capitalised_costs", "Capex / (capex + opex)", div(b["capex"], b["capex"] + opex), "pct"),
        (
            "one_off_share",
            "|Other non-operating| / pre-tax",
            div(b["other_non_operating"].abs(), b["pretax_income"].abs()),
            "pct",
        ),
    ]
    one_offs = div(b["other_non_operating"].abs(), b["pretax_income"].abs())
    return _table(
        Book(b.wide.iloc[:, 1:]),
        [(c, lbl, v.iloc[1:], u) for c, lbl, v, u in rows],
        {"one_off_years": int((one_offs > 0.10).sum())},
    )


# ── S-008 dividend safety / S-010 refinancing ────────────────────────────────


@_register(
    "S-008",
    "Dividend safety score",
    "0–100 from payout on NI (≤ 50 % best, ≥ 100 % worst), payout on FCF (≤ 50 % / ≥ 100 %), "
    "net debt / EBITDA (≤ 1 / ≥ 4), dividend growth streak (≥ 10 years best) and earnings "
    "volatility (std of NI growth, ≤ 10 % / ≥ 50 %), equally weighted.",
)
def dividend_safety(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, 12)
    if not (b.z("dividends_paid") > 0).iloc[-1]:
        raise InsufficientDataError("The company paid no dividend in the latest fiscal year")
    shares = b["shares_diluted_wavg"].fillna(b["shares_outstanding"])
    dps = div(b.z("dividends_paid"), shares)
    streak = 0
    for change in reversed(dps.diff().iloc[1:].tolist()):
        if not (np.isfinite(change) and change > 0):
            break
        streak += 1
    parts = {
        "payout_ni": (_f(div(b.z("dividends_paid"), b["net_income"]).iloc[-1]), 1.0, 0.5),
        "payout_fcf": (_f(div(b.z("dividends_paid"), b["fcf"]).iloc[-1]), 1.0, 0.5),
        "nd_ebitda": (_f(div(b.debt() - b.cash(), b["ebitda"]).iloc[-1]), 4.0, 1.0),
        "growth_streak": (float(streak), 0.0, 10.0),
        "earnings_volatility": (_f(growth(b["net_income"]).std()), 0.5, 0.1),
    }
    scores = {
        k: (linear_score(v, bad, good) if v is not None else None)
        for k, (v, bad, good) in parts.items()
    }
    available = [s for s in scores.values() if s is not None]
    return AnalysisOutput(
        "table",
        {
            "score": float(np.mean(available)) if available else None,
            "components": {k: {"value": parts[k][0], "score": scores[k]} for k in parts},
            "dps_increase_streak": streak,
            "period": b.periods[-1],
        },
    )


@_register(
    "S-010",
    "Interest-rate & refinancing exposure",
    "Debt maturing in the next 24 months (XBRL years 1–2) against cash + short-term investments "
    "+ one year of FCF; coverage ratio and share of total debt maturing.",
)
def refinancing(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, 2)
    due = b.z("debt_maturity_y1") + b.z("debt_maturity_y2")
    if not due.iloc[-1] > 0:
        raise InsufficientDataError("No debt maturities tagged (or nothing due within 24 months)")
    resources = b.cash() + b.z("fcf").clip(lower=0)
    return AnalysisOutput(
        "table",
        {
            "period": b.periods[-1],
            "due_24m": float(due.iloc[-1]),
            "liquid_resources": float(resources.iloc[-1]),
            "coverage": _f((resources / due).iloc[-1]),
            "share_of_debt": _f(div(due, b.debt()).iloc[-1]),
            "floating_rate_share": None,
        },
        ("Floating-rate share needs debt notes parsing (not in companyfacts).",),
    )


# ── S-005 screen flags ───────────────────────────────────────────────────────


@_register(
    "S-005",
    "Screen flags",
    "Descriptive flags with their rule: quality (ROE > 15 %, D/E < 0.5, gross margin > 40 %), "
    "cash quality (FCF/NI > 1.2); value (EV/FCF < 15, TTM EV from the stored price).",
    FyParams,
    {"fundamentals", "prices"},
    "1.1.0",
)
def screen_flags(inputs: AnalysisInputs, params: FyParams) -> AnalysisOutput:
    b = _book(inputs, 3)
    roe = _f(div(b["net_income"], b.avg("total_equity")).iloc[-1])
    de = _f(div(b.debt(), b["total_equity"]).iloc[-1])
    gm = _f(div(b["gross_profit"], b["revenue"]).iloc[-1])
    fcf_ni = _f(div(b["fcf"], b["net_income"]).iloc[-1])
    ev_fcf = None
    if isinstance(inputs.data.get("prices"), pd.DataFrame):
        from fda_core.analyses.valuation import NoParams, current_multiples

        try:
            ev_fcf = _f(current_multiples(inputs, NoParams()).payload["multiples"].get("ev_fcf"))
        except InsufficientDataError:
            ev_fcf = None

    def flag(value: float | None, ok: bool) -> bool | None:
        return None if value is None else ok

    flags = [
        {
            "flag": "high_roe",
            "rule": "ROE > 15 %",
            "value": roe,
            "on": flag(roe, (roe or 0) > 0.15),
        },
        {
            "flag": "low_leverage",
            "rule": "D/E < 0.5",
            "value": de,
            "on": flag(de, (de if de is not None else 9) < 0.5),
        },
        {
            "flag": "high_gross_margin",
            "rule": "Gross margin > 40 %",
            "value": gm,
            "on": flag(gm, (gm or 0) > 0.40),
        },
        {
            "flag": "cash_quality",
            "rule": "FCF / NI > 1.2",
            "value": fcf_ni,
            "on": flag(fcf_ni, (fcf_ni or 0) > 1.2),
        },
        {
            "flag": "value",
            "rule": "EV / FCF < 15",
            "value": ev_fcf,
            "on": flag(ev_fcf, (ev_fcf if ev_fcf is not None else 99) < 15),
        },
    ]
    return AnalysisOutput("table", {"period": b.periods[-1], "flags": flags})
