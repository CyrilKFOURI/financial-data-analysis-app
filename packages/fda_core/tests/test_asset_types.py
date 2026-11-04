# ruff: noqa: E501 — the 40-row fixture reads best one symbol per line
"""ASSET_TYPES.md §7 AC-1: the 40-symbol classification fixture (≥ 39/40 correct, rest low)."""

import pytest

from fda_core.asset_types import (
    TAXONOMY,
    Classification,
    InstrumentFacts,
    apply_overrides,
    classify,
    conventions,
)

E, ETF, C, IX = "equity", "etf", "crypto", "index"

# (facts as yfinance/CoinGecko report them, expected class, expected subclass)
FIXTURE: list[tuple[InstrumentFacts, str, str]] = [
    (InstrumentFacts("AAPL", "Apple Inc.", E, "common_stock", "Technology", "Consumer Electronics", "US"), E, "common_stock"),
    (InstrumentFacts("MC.PA", "LVMH Moët Hennessy", E, "common_stock", "Consumer Cyclical", "Luxury Goods", "FR"), E, "common_stock"),
    (InstrumentFacts("ASML.AS", "ASML Holding N.V.", E, "common_stock", "Technology", "Semiconductor Equipment & Materials", "NL"), E, "common_stock"),
    (InstrumentFacts("SAP.DE", "SAP SE", E, "common_stock", "Technology", "Software - Application", "DE"), E, "common_stock"),
    (InstrumentFacts("O", "Realty Income Corporation", E, "common_stock", "Real Estate", "REIT - Retail", "US"), E, "reit"),
    (InstrumentFacts("PLD", "Prologis, Inc.", E, "common_stock", "Real Estate", "REIT - Industrial", "US"), E, "reit"),
    (InstrumentFacts("URW.AS", "Unibail-Rodamco-Westfield SE", E, "common_stock", "Real Estate", "REIT - Retail", "FR"), E, "reit"),
    (InstrumentFacts("JPM", "JPMorgan Chase & Co.", E, "common_stock", "Financial Services", "Banks - Diversified", "US"), E, "common_stock"),
    (InstrumentFacts("BNP.PA", "BNP Paribas SA", E, "common_stock", "Financial Services", "Banks - Regional", "FR"), E, "common_stock"),
    (InstrumentFacts("ALV.DE", "Allianz SE", E, "common_stock", "Financial Services", "Insurance - Diversified", "DE"), E, "common_stock"),
    (InstrumentFacts("TSM", "Taiwan Semiconductor Manufacturing Company Limited ADR", E, "common_stock", "Technology", "Semiconductors", "TW"), E, "adr_gdr"),
    (InstrumentFacts("BABA", "Alibaba Group Holding Limited ADS", E, "common_stock", "Consumer Cyclical", "Internet Retail", "CN"), E, "adr_gdr"),
    (InstrumentFacts("BAC-PL", "Bank of America Corporation Preferred", E, "common_stock", "Financial Services", "Banks - Diversified", "US"), E, "preferred_stock"),
    (InstrumentFacts("SPY", "SPDR S&P 500 ETF Trust", ETF, "equity_etf", profile={"category": "Large Blend"}), ETF, "equity_etf"),
    (InstrumentFacts("IWDA.AS", "iShares Core MSCI World UCITS ETF", ETF, "equity_etf", profile={"category": "Global Large-Cap Blend Equity"}), ETF, "equity_etf"),
    (InstrumentFacts("CW8.PA", "Amundi MSCI World UCITS ETF", ETF, "equity_etf"), ETF, "equity_etf"),
    (InstrumentFacts("AGG", "iShares Core U.S. Aggregate Bond ETF", ETF, "equity_etf", profile={"category": "Intermediate Core Bond"}), ETF, "bond_etf"),
    (InstrumentFacts("TLT", "iShares 20+ Year Treasury Bond ETF", ETF, "equity_etf", profile={"category": "Long Government"}), ETF, "bond_etf"),
    (InstrumentFacts("GLD", "SPDR Gold Shares", ETF, "equity_etf", profile={"category": "Commodities Focused"}), ETF, "commodity_etf"),
    (InstrumentFacts("TQQQ", "ProShares UltraPro QQQ", ETF, "equity_etf", profile={"category": "Trading--Leveraged Equity"}), ETF, "leveraged_etf"),
    (InstrumentFacts("SSO", "ProShares Ultra S&P500", ETF, "equity_etf", profile={"category": "Trading--Leveraged Equity"}), ETF, "leveraged_etf"),
    (InstrumentFacts("SQQQ", "ProShares UltraPro Short QQQ", ETF, "equity_etf", profile={"category": "Trading--Inverse Equity"}), ETF, "inverse_etf"),
    (InstrumentFacts("SH", "ProShares Short S&P500", ETF, "equity_etf", profile={"category": "Trading--Inverse Equity"}), ETF, "inverse_etf"),
    (InstrumentFacts("IWDE.L", "iShares MSCI World EUR Hedged UCITS ETF", ETF, "equity_etf"), ETF, "currency_hedged_etf"),
    (InstrumentFacts("ARKK", "ARK Innovation ETF (Active)", ETF, "equity_etf", profile={"category": "Mid-Cap Growth"}), ETF, "active_etf"),
    (InstrumentFacts("VFIAX", "Vanguard 500 Index Admiral", "fund", "mutual_fund"), "fund", "mutual_fund"),
    (InstrumentFacts("^GSPC", "S&P 500", IX, "equity_index"), IX, "equity_index"),
    (InstrumentFacts("^STOXX", "STOXX Europe 600", IX, "equity_index"), IX, "equity_index"),
    (InstrumentFacts("^VIX", "CBOE Volatility Index", IX, "equity_index"), IX, "volatility_index"),
    (InstrumentFacts("^TNX", "Treasury Yield 10 Years", IX, "equity_index"), IX, "rate_index"),
    (InstrumentFacts("BTC-USD", "Bitcoin USD", C, "coin"), C, "coin"),
    (InstrumentFacts("ETH-USD", "Ethereum USD", C, "coin"), C, "coin"),
    (InstrumentFacts("LINK-USD", "Chainlink USD", C, "coin"), C, "token"),
    (InstrumentFacts("UNI-USD", "Uniswap USD", C, "coin"), C, "token"),
    (InstrumentFacts("USDT-USD", "Tether USDt USD", C, "coin"), C, "stablecoin"),
    (InstrumentFacts("USDC-USD", "USD Coin USD", C, "coin"), C, "stablecoin"),
    (InstrumentFacts("EURUSD=X", "EUR/USD", "fx", "spot_pair"), "fx", "spot_pair"),
    (InstrumentFacts("CL=F", "Crude Oil Nov 26", "commodity", "continuous_future"), "commodity", "continuous_future"),
    (InstrumentFacts("DGS10", "10-Year Treasury constant maturity", "rate"), "rate", "yield"),
    (InstrumentFacts("CPIAUCSL", "CPI, all urban consumers", "macro"), "macro", "inflation"),
]  # fmt: skip


def test_fixture_has_forty_symbols_and_valid_taxonomy() -> None:
    assert len(FIXTURE) == 40
    for _, cls, sub in FIXTURE:
        assert sub in TAXONOMY[cls]


def test_classification_accuracy_on_the_40_symbol_fixture() -> None:
    wrong = []
    for facts, cls, sub in FIXTURE:
        result = classify(facts)
        if (result.asset_class, result.asset_subclass) != (cls, sub):
            wrong.append((facts.symbol, result))
    assert len(wrong) <= 1, wrong
    assert all(result.confidence == "low" for _, result in wrong)


@pytest.mark.parametrize(
    ("symbol", "factor"), [("TQQQ", 3.0), ("SSO", 2.0), ("SQQQ", -3.0), ("SH", -1.0)]
)
def test_leverage_factor_and_daily_reset_flag(symbol: str, factor: float) -> None:
    facts = next(f for f, _, _ in FIXTURE if f.symbol == symbol)
    result = classify(facts)
    assert result.leverage_factor == factor
    assert "daily_reset" in result.flags
    assert "decay" in conventions(result).caveats[0]


def test_bank_and_insurer_flags_are_company_flags() -> None:
    jpm = classify(next(f for f, _, _ in FIXTURE if f.symbol == "JPM"))
    alv = classify(next(f for f, _, _ in FIXTURE if f.symbol == "ALV.DE"))
    assert jpm.flags == ("bank",)
    assert alv.flags == ("insurer",)


def test_symbol_grammar_without_provider_type() -> None:
    assert classify(InstrumentFacts("^FCHI")).asset_class == "index"
    assert classify(InstrumentFacts("GBPUSD=X")).asset_class == "fx"
    assert classify(InstrumentFacts("SOL-USD")).asset_subclass == "coin"
    assert classify(InstrumentFacts("XYZ")).confidence == "low"


def test_overrides_win_and_are_validated() -> None:
    base = classify(next(f for f, _, _ in FIXTURE if f.symbol == "O"))
    overridden = apply_overrides(base, {"asset_subclass": "common_stock"})
    assert (overridden.asset_subclass, overridden.confidence) == ("common_stock", "override")
    assert "rules said reit" in overridden.reason
    with pytest.raises(ValueError, match="taxonomy"):
        apply_overrides(base, {"asset_subclass": "stablecoin"})


@pytest.mark.parametrize(
    ("facts", "annualisation", "price_field", "has_volume"),
    [
        (InstrumentFacts("AAPL", "Apple", E, "common_stock"), 252, "adj_close", True),
        (InstrumentFacts("BTC-USD", "Bitcoin", C, "coin"), 365, "close", True),
        (InstrumentFacts("EURUSD=X", "EUR/USD", "fx", "spot_pair"), 260, "close", False),
        (InstrumentFacts("DGS10", "10y", "rate"), None, "level", False),
    ],
)
def test_conventions_follow_the_type(
    facts: InstrumentFacts, annualisation: int | None, price_field: str, has_volume: bool
) -> None:
    conv = conventions(classify(facts))
    assert (conv.annualisation, conv.price_field, conv.has_volume) == (
        annualisation,
        price_field,
        has_volume,
    )


def test_benchmark_defaults_to_the_listing_currency_market() -> None:
    lvmh = Classification("equity", "common_stock", "provider", "")
    assert conventions(lvmh, "EUR").benchmark == "^STOXX"
    assert conventions(lvmh, "USD").benchmark == "SPY"
