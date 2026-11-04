"""Asset-type intelligence (ASSET_TYPES.md §2-3): deterministic, explainable classification and
the conventions every analysis inherits from the type. Pure functions, no I/O.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

AssetClass = Literal[
    "equity", "etf", "fund", "index", "crypto", "fx", "commodity", "rate", "macro", "option",
    "bond",
]  # fmt: skip
Confidence = Literal["override", "rule", "provider", "low"]

TAXONOMY: dict[str, tuple[str, ...]] = {
    "equity": ("common_stock", "preferred_stock", "adr_gdr", "reit", "spac_shell", "synthetic"),
    "etf": (
        "equity_etf", "bond_etf", "commodity_etf", "leveraged_etf", "inverse_etf",
        "currency_hedged_etf", "thematic_etf", "active_etf",
    ),
    "fund": ("mutual_fund", "money_market"),
    "index": ("equity_index", "volatility_index", "rate_index"),
    "crypto": ("coin", "token", "stablecoin"),
    "fx": ("spot_pair",),
    "commodity": ("continuous_future", "contract"),
    "rate": ("yield", "spread", "policy_rate"),
    "macro": ("inflation", "activity"),
    "option": ("equity_option", "index_option"),
    "bond": ("government", "corporate"),
}  # fmt: skip

STABLECOINS = frozenset(
    {
        "USDT",
        "USDC",
        "DAI",
        "TUSD",
        "USDP",
        "FDUSD",
        "PYUSD",
        "USDE",
        "BUSD",
        "GUSD",
        "EURC",
        "USDS",
    }
)
# Native layer-1 coins; any other crypto pair is treated as a token.
COINS = frozenset(
    {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "LTC", "BCH", "DOT", "AVAX", "TRX", "XLM",
        "ATOM", "XMR", "ETC", "NEAR", "ALGO", "TON", "SUI", "APT", "HBAR", "ICP", "BNB", "KAS",
    }
)  # fmt: skip
VOLATILITY_INDICES = frozenset({"^VIX", "^VXN", "^VSTOXX", "^V2TX", "^VVIX", "^VIX3M", "^RVX"})
RATE_INDICES = frozenset({"^TNX", "^IRX", "^FVX", "^TYX"})
RATE_SERIES = {"DGS": "yield", "DTB": "yield", "T10Y": "spread", "T5YIE": "spread"}

_LEVERAGE_RE = re.compile(r"(?<![\d.])([2-5])(?:\.0)?\s?[xX×](?![a-z])|\b(Ultra ?Pro|Ultra)\b")
_INVERSE_RE = re.compile(r"\b(Inverse|Short|Bear)\b", re.IGNORECASE)
_BOND_RE = re.compile(r"\b(Bond|Treasury|Treasuries|Aggregate|Corporate|Gilt|Fixed Income)\b", re.I)
_COMMODITY_RE = re.compile(r"\b(Gold|Silver|Oil|Commodit\w*|Physical|Platinum|Copper)\b", re.I)
_HEDGED_RE = re.compile(r"\bHedged\b", re.IGNORECASE)
_ACTIVE_RE = re.compile(r"\bActive\b", re.IGNORECASE)
_ADR_RE = re.compile(r"\b(ADR|ADS|American Depositary)\b")
_PREFERRED_RE = re.compile(r"(-P[A-Z]?$|\bPreferred\b|\bPref\b|Depositary Shares)")
_SPAC_RE = re.compile(r"\b(Acquisition Corp|Blank Check)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Classification:
    asset_class: str
    asset_subclass: str
    confidence: Confidence
    reason: str
    leverage_factor: float | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InstrumentFacts:
    """What the providers told us; every field is optional."""

    symbol: str
    name: str = ""
    provider_class: str | None = None
    provider_subclass: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    profile: Mapping[str, Any] = field(default_factory=dict)


def _base(symbol: str) -> str:
    return symbol.split("-")[0].upper()


def _text(facts: InstrumentFacts) -> str:
    return " ".join(
        str(part)
        for part in (facts.name, facts.profile.get("category"), facts.profile.get("longName"))
        if part
    )


def _grammar_class(symbol: str) -> tuple[str, str, str] | None:
    """Symbol grammar (§2.2 step 2) when no provider type is available."""
    upper = symbol.upper()
    if upper.startswith("^"):
        return "index", "equity_index", "symbol starts with ^"
    if upper.endswith("=X"):
        return "fx", "spot_pair", "symbol ends with =X"
    if upper.endswith("=F"):
        return "commodity", "continuous_future", "symbol ends with =F"
    if re.fullmatch(r"[A-Z0-9]{2,10}-(USD|EUR|USDT|GBP)", upper):
        return "crypto", "coin", "symbol is <coin>-<quote>"
    return None


def _classify_etf(facts: InstrumentFacts) -> Classification:
    text = _text(facts)
    category = str(facts.profile.get("category") or "")
    leverage = _LEVERAGE_RE.search(text)
    inverse = bool(_INVERSE_RE.search(text)) or "Inverse" in category
    if leverage or "Trading--Leveraged" in category:
        factor = 2.0
        if leverage and leverage.group(1):
            factor = float(leverage.group(1))
        elif leverage and leverage.group(2) and "Pro" in leverage.group(2):
            factor = 3.0
        if inverse:
            return Classification(
                "etf", "inverse_etf", "rule", f"inverse wording in “{text}”", -factor,
                ("daily_reset",),
            )  # fmt: skip
        return Classification(
            "etf", "leveraged_etf", "rule", f"leverage wording in “{text}”", factor,
            ("daily_reset",),
        )  # fmt: skip
    if inverse:
        return Classification(
            "etf", "inverse_etf", "rule", f"inverse wording in “{text}”", -1.0, ("daily_reset",)
        )
    if _HEDGED_RE.search(text):
        return Classification("etf", "currency_hedged_etf", "rule", "“Hedged” in name")
    if _BOND_RE.search(text):
        return Classification("etf", "bond_etf", "rule", f"fixed-income wording in “{text}”")
    if _COMMODITY_RE.search(text):
        return Classification("etf", "commodity_etf", "rule", f"commodity wording in “{text}”")
    if _ACTIVE_RE.search(text):
        return Classification("etf", "active_etf", "rule", "“Active” in name")
    return Classification("etf", "equity_etf", "provider", "provider type ETF, equity by default")


def _classify_equity(facts: InstrumentFacts) -> Classification:
    industry = facts.industry or ""
    sector = facts.sector or ""
    flags: list[str] = []
    if "Bank" in industry:
        flags.append("bank")
    if "Insurance" in industry:
        flags.append("insurer")
    if _PREFERRED_RE.search(facts.symbol.upper()) or _PREFERRED_RE.search(facts.name):
        return Classification("equity", "preferred_stock", "rule", "preferred share grammar")
    if industry.startswith("REIT") or (sector == "Real Estate" and "REIT" in industry):
        return Classification(
            "equity", "reit", "rule", f"industry “{industry}”", None, tuple(flags)
        )
    if _ADR_RE.search(facts.name):
        return Classification("equity", "adr_gdr", "rule", "ADR/ADS in name", None, tuple(flags))
    if _SPAC_RE.search(facts.name) and not industry:
        return Classification("equity", "spac_shell", "rule", "blank-check wording, no industry")
    reason = "provider type EQUITY"
    if flags:
        reason += f"; {'/'.join(flags)} flag from industry “{industry}”"
    return Classification("equity", "common_stock", "provider", reason, None, tuple(flags))


def _classify_crypto(facts: InstrumentFacts) -> Classification:
    base = _base(facts.symbol)
    if base in STABLECOINS:
        return Classification("crypto", "stablecoin", "rule", f"{base} is in the stablecoin list")
    if base in COINS:
        return Classification("crypto", "coin", "rule", f"{base} is a native layer-1 coin")
    categories = [str(c) for c in facts.profile.get("categories") or []]
    if any("Stablecoin" in c for c in categories):
        return Classification("crypto", "stablecoin", "rule", "CoinGecko category Stablecoins")
    return Classification("crypto", "token", "rule", f"{base} is not a native coin")


def _classify_index(facts: InstrumentFacts) -> Classification:
    upper = facts.symbol.upper()
    if upper in VOLATILITY_INDICES or "Volatility" in facts.name:
        return Classification("index", "volatility_index", "rule", "volatility index")
    if upper in RATE_INDICES or "Treasury" in facts.name or "Yield" in facts.name:
        return Classification("index", "rate_index", "rule", "rate index")
    return Classification("index", "equity_index", "provider", "provider type INDEX")


def _classify_series(facts: InstrumentFacts) -> Classification:
    code = facts.symbol.upper()
    if code.startswith(("CPI", "PCE")):
        return Classification("macro", "inflation", "rule", "inflation series code")
    for prefix, subclass in RATE_SERIES.items():
        if code.startswith(prefix):
            return Classification("rate", subclass, "rule", f"rate series code {prefix}…")
    if code in {"ESTR", "DFF", "SOFR"}:
        return Classification("rate", "policy_rate", "rule", "overnight reference rate")
    return Classification("macro", "activity", "low", "unknown series code")


def classify(facts: InstrumentFacts) -> Classification:
    """Provider type → symbol grammar → metadata rules (§2.2 steps 1-3). Overrides (step 4)
    are applied by `apply_overrides` on top."""
    provider_class = facts.provider_class
    if provider_class is None:
        grammar = _grammar_class(facts.symbol)
        if grammar is None:
            return Classification("equity", "common_stock", "low", "no provider type, no grammar")
        provider_class = grammar[0]
    match provider_class:
        case "etf":
            return _classify_etf(facts)
        case "equity":
            if facts.provider_subclass == "synthetic":
                return Classification("equity", "synthetic", "provider", "synthetic test asset")
            return _classify_equity(facts)
        case "crypto":
            return _classify_crypto(facts)
        case "index":
            return _classify_index(facts)
        case "fund":
            subclass = facts.provider_subclass or "mutual_fund"
            return Classification("fund", subclass, "provider", f"provider type {subclass}")
        case "fx":
            return Classification("fx", "spot_pair", "provider", "provider type CURRENCY")
        case "commodity":
            return Classification("commodity", "continuous_future", "provider", "provider FUTURE")
        case "rate" | "macro":
            return _classify_series(facts)
        case _:
            return Classification(provider_class, facts.provider_subclass or "", "low", "unmapped")


def apply_overrides(base: Classification, overrides: Mapping[str, str]) -> Classification:
    """User / seed overrides win over rules; the reason records what was overridden."""
    if not overrides:
        return base
    asset_class = overrides.get("asset_class", base.asset_class)
    subclass = overrides.get("asset_subclass", base.asset_subclass)
    if subclass not in TAXONOMY.get(asset_class, ()):
        raise ValueError(f"{asset_class}/{subclass} is not in the taxonomy")
    leverage = overrides.get("leverage_factor")
    return Classification(
        asset_class,
        subclass,
        "override",
        f"override ({', '.join(sorted(overrides))}); rules said {base.asset_subclass}",
        float(leverage) if leverage is not None else base.leverage_factor,
        base.flags,
    )


# ── conventions (§3) ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Conventions:
    annualisation: int | None
    calendar: Literal["exchange", "24/7", "24/5", "business_days"]
    price_field: Literal["adj_close", "close", "level"]
    has_volume: bool
    return_kind: Literal["simple", "level_change"]
    benchmark: str | None
    caveats: tuple[str, ...] = ()


def conventions(classification: Classification, currency: str | None = None) -> Conventions:
    cls, sub = classification.asset_class, classification.asset_subclass
    caveats: list[str] = []
    if "daily_reset" in classification.flags:
        caveats.append(
            f"Daily-reset {classification.leverage_factor:+g}× product: multi-day returns drift "
            "from the stated multiple (volatility decay)."
        )
    match cls:
        case "crypto":
            if sub == "stablecoin":
                caveats.append("Stablecoin: analyse the peg deviation, not returns.")
            return Conventions(
                365, "24/7", "close", True, "simple", None if sub == "coin" else "BTC-USD",
                (*caveats, "Volume is exchange-dependent and indicative only."),
            )  # fmt: skip
        case "fx":
            return Conventions(260, "24/5", "close", False, "simple", "DX-Y.NYB", tuple(caveats))
        case "commodity":
            caveats.append("Continuous future: roll gaps are embedded in the series.")
            return Conventions(252, "exchange", "close", True, "simple", "DBC", tuple(caveats))
        case "index":
            caveats.append("Indices are not investable: use a tracker for portfolios.")
            return Conventions(252, "exchange", "close", False, "simple", None, tuple(caveats))
        case "rate" | "macro":
            return Conventions(None, "business_days", "level", False, "level_change", None)
        case _:
            benchmark = {"EUR": "^STOXX", "GBP": "^FTSE", "JPY": "^N225"}.get(currency or "", "SPY")
            if cls == "equity" and sub == "reit":
                caveats.append("REIT: FFO/AFFO matter more than EPS.")
            if sub == "adr_gdr":
                caveats.append("Depositary receipt: returns include an FX layer.")
            if sub == "synthetic":
                caveats.append("Synthetic test data, not a real instrument.")
            return Conventions(
                252, "exchange", "adj_close", True, "simple", benchmark, tuple(caveats)
            )
