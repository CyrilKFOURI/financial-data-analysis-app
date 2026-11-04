"""Real-provider adapters replayed from recorded responses (fixtures/, recorded 2026-10-23).

No test here touches the network; `-m live` runs the opt-in smoke tests at the bottom.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pandas as pd
import pytest
from sqlalchemy import Engine, select

from fda_data.chain import NoProviderError, ProviderChain, SeriesChain
from fda_data.ingest import ingest_daily_prices, ingest_series
from fda_data.lake import PriceLake
from fda_data.providers.base import RateLimitedError, SymbolNotFoundError
from fda_data.providers.coingecko import CoinGeckoProvider, base_asset
from fda_data.providers.ecb import EcbProvider
from fda_data.providers.fred import FredProvider
from fda_data.providers.http import make_client
from fda_data.providers.yahoo import YahooProvider
from fda_data.ratelimit import TokenBucket
from fda_data.schema import ingestion_jobs, instruments, series_catalog

FIXTURES = Path(__file__).parent / "fixtures"
INFO: dict[str, dict[str, Any]] = json.loads(
    (FIXTURES / "yahoo" / "info.json").read_text(encoding="utf-8")
)


class RecordedTicker:
    """Stands in for `yfinance.Ticker` with recorded `history()` / `get_info()` output."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def get_info(self) -> dict[str, Any]:
        return INFO.get(self.symbol, {"trailingPegRatio": None})

    def history(self, **kwargs: Any) -> pd.DataFrame:
        path = FIXTURES / "yahoo" / f"{self.symbol}_history.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path, index_col="Date")
        tz = "UTC" if self.symbol.endswith("-USD") else "America/New_York"
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(tz)
        start, end = pd.Timestamp(kwargs["start"], tz=tz), pd.Timestamp(kwargs["end"], tz=tz)
        return frame.loc[(frame.index >= start) & (frame.index < end)]


class ThrottledTicker(RecordedTicker):
    def get_info(self) -> dict[str, Any]:
        raise type("YFRateLimitError", (Exception,), {})("Too Many Requests. Rate limited.")


def recorded_search(query: str) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = json.loads(
        (FIXTURES / "yahoo" / f"search_{query}.json").read_text(encoding="utf-8")
    )
    return quotes


def yahoo() -> YahooProvider:
    return YahooProvider(ticker_factory=RecordedTicker, search_fn=recorded_search)


def replay(routes: dict[str, str], status: int = 200) -> httpx.Client:
    """HTTP client answering from recorded bodies, keyed by a URL substring."""

    def handler(request: httpx.Request) -> httpx.Response:
        for fragment, filename in routes.items():
            if fragment in str(request.url):
                body = (FIXTURES / "http" / filename).read_bytes()
                return httpx.Response(status, content=body)
        return httpx.Response(404)

    return make_client(httpx.MockTransport(handler))


# ── yfinance ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("symbol", "asset_class", "asset_subclass", "currency"),
    [
        ("AAPL", "equity", "common_stock", "USD"),
        ("MC.PA", "equity", "common_stock", "EUR"),
        ("SPY", "etf", "equity_etf", "USD"),
        ("VFIAX", "fund", "mutual_fund", "USD"),
        ("^GSPC", "index", "equity_index", "USD"),
        ("BTC-USD", "crypto", "coin", "USD"),
        ("EURUSD=X", "fx", "spot_pair", "USD"),
        ("GC=F", "commodity", "continuous_future", "USD"),
    ],
)
def test_yahoo_profile_maps_quote_types(
    symbol: str, asset_class: str, asset_subclass: str, currency: str
) -> None:
    profile = yahoo().profile(symbol)
    assert (profile.asset_class, profile.asset_subclass) == (asset_class, asset_subclass)
    assert profile.currency == currency
    assert profile.extras["quoteType"] == INFO[symbol]["quoteType"]


def test_yahoo_profile_keeps_inputs_for_type_detection() -> None:
    spy = yahoo().profile("SPY")
    assert spy.extras["legalType"] == "Exchange Traded Fund"
    assert spy.extras["category"] == "Large Blend"
    aapl = yahoo().profile("aapl")
    assert (aapl.country, aapl.sector, aapl.timezone) == ("US", "Technology", "America/New_York")


def test_yahoo_unknown_symbol_and_rate_limit_are_typed_errors() -> None:
    with pytest.raises(SymbolNotFoundError):
        yahoo().profile("NOPE")
    with pytest.raises(RateLimitedError):
        YahooProvider(ticker_factory=ThrottledTicker).profile("AAPL")
    assert not yahoo().supports("FAKE")
    assert not yahoo().supports("bad symbol")


def test_yahoo_bars_keep_raw_and_adjusted_close() -> None:
    bars = yahoo().daily_bars("AAPL", date(2020, 8, 3), date(2020, 9, 4))
    assert list(bars.columns) == [
        "session_date", "open", "high", "low", "close", "adj_close", "volume",
    ]  # fmt: skip
    assert bars["session_date"].iloc[0] == date(2020, 8, 3)
    assert bars["session_date"].iloc[-1] == date(2020, 9, 4)  # inclusive end
    assert (bars["adj_close"] < bars["close"]).all()  # later dividends lower the adjusted series


def test_yahoo_actions_extract_dividend_and_split() -> None:
    provider = yahoo()
    provider.profile("AAPL")
    actions = provider.actions("AAPL", date(2020, 8, 3), date(2020, 9, 4))
    dividend, split = actions.to_dict("records")
    assert (dividend["ex_date"], dividend["action"], dividend["currency"]) == (
        date(2020, 8, 7), "dividend", "USD",
    )  # fmt: skip
    assert dividend["amount"] == pytest.approx(0.205)
    assert pd.isna(dividend["ratio"])
    assert (split["ex_date"], split["action"], split["ratio"]) == (date(2020, 8, 31), "split", 4.0)


def test_yahoo_crypto_sessions_include_weekends() -> None:
    bars = yahoo().daily_bars("BTC-USD", date(2025, 1, 1), date(2025, 1, 7))
    assert len(bars) == 7


def test_yahoo_search_keeps_supported_types_only() -> None:
    results = yahoo().search("apple")
    assert results[0].symbol == "AAPL"
    assert results[0].asset_class == "equity"
    assert all(r.asset_class for r in results)


# ── FRED / ECB / CoinGecko ────────────────────────────────────────────────────


def test_fred_drops_holiday_gaps_instead_of_filling() -> None:
    provider = FredProvider(replay({"fredgraph": "fred_DTB3.csv"}))
    frame = provider.series("DTB3", date(2024, 12, 20), date(2025, 1, 10))
    assert date(2024, 12, 25) not in set(frame["date"])
    assert frame["value"].iloc[0] == pytest.approx(4.23)
    assert provider.series_info("DTB3").unit == "percent"


def test_ecb_estr_and_reference_fx() -> None:
    client = replay({"/EST/": "ecb_ESTR.csv", "/EXR/": "ecb_EURUSD.csv"})
    provider = EcbProvider(client)
    estr = provider.series("ESTR", date(2025, 1, 2), date(2025, 1, 10))
    assert estr["value"].iloc[0] == pytest.approx(2.92)
    fx = provider.series("ECB_EURUSD", date(2025, 1, 2), date(2025, 1, 10))
    assert fx["value"].iloc[0] == pytest.approx(1.0321)
    assert provider.series_info("ECB_EURJPY").provider_key == "EXR/D.JPY.EUR.SP00.A"
    assert not provider.supports_series("DTB3")


def test_series_chain_routes_ecb_codes_before_fred() -> None:
    chain = SeriesChain([EcbProvider(replay({})), FredProvider(replay({}))])
    assert chain.info("ESTR").source == "ecb"
    assert chain.info("DGS10").source == "fred"
    assert not chain.can_serve("bad-code")


def test_http_429_is_a_rate_limit() -> None:
    provider = FredProvider(replay({"fredgraph": "fred_DTB3.csv"}, status=429))
    with pytest.raises(RateLimitedError):
        provider.series("DTB3", date(2025, 1, 1), date(2025, 1, 10))


def test_coingecko_picks_the_real_coin_over_wrapped_tokens() -> None:
    client = replay({"/search": "coingecko_search_BTC.json", "/coins/": "coingecko_bitcoin.json"})
    provider = CoinGeckoProvider(client, TokenBucket(rate_per_second=100, burst=10))
    metadata = provider.crypto_metadata("BTC-USD")
    assert metadata["coingecko_id"] == "bitcoin"
    assert metadata["max_supply"] == 21_000_000
    assert metadata["market_cap_rank"] == 1
    assert base_asset("ETH-EUR") == "ETH"


# ── ingestion with real adapters ──────────────────────────────────────────────


def test_ingest_aapl_stores_bars_actions_and_explains_the_split(
    engine: Engine, lake: PriceLake
) -> None:
    chain = ProviderChain([yahoo()])
    # Window ending 2020-09-04 with a 10-year default history (fixture covers Aug-Sep 2020).
    result = ingest_daily_prices(engine, lake, chain, "AAPL", today=date(2020, 9, 4))
    assert result.status == "succeeded"
    assert result.source == "yfinance"
    assert result.qc["actions"] == 2
    assert result.qc["flagged_spikes"] == []

    stored = lake.read_actions(result.instrument_id or 0)
    assert set(stored["action"]) == {"dividend", "split"}
    with engine.connect() as conn:
        datasets = set(conn.execute(select(series_catalog.c.dataset)).scalars())
        profile = conn.execute(select(instruments.c.profile)).scalar_one()
    assert datasets == {"prices_daily", "actions"}
    assert profile["fullExchangeName"] == "NasdaqGS"


def test_ingest_crypto_attaches_coingecko_metadata(engine: Engine, lake: PriceLake) -> None:
    client = replay({"/search": "coingecko_search_BTC.json", "/coins/": "coingecko_bitcoin.json"})
    coingecko = CoinGeckoProvider(client, TokenBucket(rate_per_second=100, burst=10))
    result = ingest_daily_prices(
        engine, lake, ProviderChain([yahoo()]), "BTC-USD", today=date(2025, 1, 7),
        coingecko=coingecko,
    )  # fmt: skip
    assert result.status == "succeeded"
    with engine.connect() as conn:
        profile = conn.execute(select(instruments.c.profile)).scalar_one()
    assert profile["coingecko_id"] == "bitcoin"
    assert profile["quoteType"] == "CRYPTOCURRENCY"


def test_coingecko_failure_never_blocks_prices(engine: Engine, lake: PriceLake) -> None:
    coingecko = CoinGeckoProvider(replay({}), TokenBucket(rate_per_second=100, burst=10))
    result = ingest_daily_prices(
        engine, lake, ProviderChain([yahoo()]), "BTC-USD", today=date(2025, 1, 7),
        coingecko=coingecko,
    )  # fmt: skip
    assert result.status == "succeeded"
    assert result.rows_written == 6  # 2025-01-07 is today: still moving, not stored


def test_rate_limited_ingestion_is_retryable(engine: Engine, lake: PriceLake) -> None:
    chain = ProviderChain([YahooProvider(ticker_factory=ThrottledTicker)])
    result = ingest_daily_prices(engine, lake, chain, "AAPL", today=date(2020, 9, 4))
    assert result.status == "failed"
    assert result.retryable is True


def test_ingest_series_is_incremental(engine: Engine, lake: PriceLake) -> None:
    chain = SeriesChain([EcbProvider(replay({"/EST/": "ecb_ESTR.csv"}))])
    first = ingest_series(engine, lake, chain, "estr", today=date(2025, 1, 10))
    assert first.status == "succeeded"
    assert first.rows_written == len(lake.read_series("ESTR"))
    second = ingest_series(engine, lake, chain, "ESTR", today=date(2025, 1, 10))
    assert second.message == "Already up to date"
    with engine.connect() as conn:
        kinds = list(conn.execute(select(ingestion_jobs.c.kind)).scalars())
    assert kinds == ["series", "series"]


def test_unknown_series_fails_cleanly(engine: Engine, lake: PriceLake) -> None:
    result = ingest_series(engine, lake, SeriesChain([EcbProvider(replay({}))]), "DGS10")
    assert result.status == "failed"
    assert "No provider knows series DGS10" in result.message
    with pytest.raises(NoProviderError):
        SeriesChain([]).info("X")


# ── live smoke tests (opt-in: uv run pytest -m live) ──────────────────────────


@pytest.mark.live
def test_live_yahoo_and_reference_series() -> None:
    provider = YahooProvider()
    assert provider.profile("AAPL").asset_class == "equity"
    assert not provider.daily_bars("AAPL", date(2025, 1, 2), date(2025, 1, 10)).empty
    assert not FredProvider().series("DTB3", date(2025, 1, 2), date(2025, 1, 10)).empty
    assert not EcbProvider().series("ESTR", date(2025, 1, 2), date(2025, 1, 10)).empty
    assert CoinGeckoProvider().crypto_metadata("BTC-USD")["coingecko_id"] == "bitcoin"


def test_ken_french_daily_table_parses_percent_to_decimals_and_drops_missing() -> None:
    from fda_data.providers.kenfrench import KenFrenchProvider, codes, parse_daily

    text = (
        "This file was created by using the 202608 CRSP database.\n\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "20260828,   -0.18,   -0.25,    0.55,    0.10,    0.33,    0.01\n"
        "20260831,   -0.30,  -99.99,    0.81,   -0.25,    0.14,    0.01\n"
        "\n Annual Factors: January-December\n,Mkt-RF\n  2025,  12.0\n"
    )
    frame = parse_daily(text)
    assert list(frame.columns) == ["date", "MKTRF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(frame) == 2
    assert frame["MKTRF"].iloc[1] == pytest.approx(-0.003)
    assert frame["SMB"].isna().iloc[1]
    mom = parse_daily("Momentum\n,Mom   \n20260831,  -0.14\n\nCopyright\n")
    assert mom["MOM"].iloc[0] == pytest.approx(-0.0014)
    provider = KenFrenchProvider(replay({}))
    assert provider.supports_series("KF_EU_HML")
    assert not provider.supports_series("DGS10")
    assert codes("US")[0] == "KF_US_MKTRF"
    assert len(codes("EU")) == 7
