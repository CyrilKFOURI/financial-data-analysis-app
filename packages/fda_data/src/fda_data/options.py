"""Option chains (M21): yfinance snapshots (delayed quotes, no history from the free source),
stored per day in the lake so IV history accumulates; a synthetic chain for FAKE* symbols.

Lake layout:
- `option_chains/symbol=<SYM>/as_of=<date>/part.parquet`: one row per contract;
- `option_iv/symbol=<SYM>/part.parquet`: one row per snapshot date (30-day ATM IV, put/call
  ratios, spot), the input of IV rank and the volatility risk premium.
"""

import math
import zlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fda_core.analyses.derivatives import chain_summary, enrich_chain
from fda_core.options import Svi, bsm_price
from fda_data.lake import PriceLake, _atomic_write
from fda_data.providers.base import ProviderError

CHAIN_COLUMNS = [
    "expiry", "kind", "strike", "bid", "ask", "last", "volume", "open_interest", "provider_iv",
    "contract", "last_trade",
]  # fmt: skip
IV_SCHEMA = pa.schema(
    [("date", pa.date32()), ("spot", pa.float64()), ("atm_iv_30d", pa.float64()),
     ("put_call_volume", pa.float64()), ("put_call_oi", pa.float64()), ("contracts", pa.int64())]
)  # fmt: skip


def fetch_yahoo_chain(
    symbol: str, today: date, max_expiries: int = 40
) -> tuple[float, pd.DataFrame]:
    """Spot and every listed expiry after today (one request per expiry)."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiries = [e for e in ticker.options if date.fromisoformat(e) > today][:max_expiries]
    if not expiries:
        raise ProviderError(f"No listed options for {symbol}")
    frames, spot = [], math.nan
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        spot = float((getattr(chain, "underlying", None) or {}).get("regularMarketPrice") or spot)
        for kind, part in (("call", chain.calls), ("put", chain.puts)):
            if part.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "expiry": date.fromisoformat(expiry),
                        "kind": kind,
                        "strike": part["strike"].astype(float),
                        "bid": part["bid"].astype(float),
                        "ask": part["ask"].astype(float),
                        "last": part["lastPrice"].astype(float),
                        "volume": part["volume"].astype(float),
                        "open_interest": part["openInterest"].astype(float),
                        "provider_iv": part["impliedVolatility"].astype(float),
                        "contract": part["contractSymbol"].astype(str),
                        "last_trade": pd.to_datetime(part["lastTradeDate"], utc=True),
                    }
                )
            )
    if not math.isfinite(spot):
        raise ProviderError(f"No underlying price in the {symbol} chain")
    return spot, pd.concat(frames, ignore_index=True)


def fake_chain(symbol: str, today: date, spot: float) -> pd.DataFrame:
    """Deterministic chain priced by BSM on an SVI smile (negative skew), 6 expiries × 31
    strikes, with a 1 % bid/ask spread around the model price, for offline tests."""
    rows = []
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))  # stable across processes
    for days in (7, 14, 30, 60, 91, 182):
        t = days / 365
        smile = Svi(a=0.04 * t * 0.5, b=0.1 * math.sqrt(t), rho=-0.6, m=0.0, s=0.2)
        for strike in np.linspace(0.7, 1.3, 31) * spot:
            iv = float(smile.iv(np.array([math.log(strike / spot)]), t)[0])
            for kind in ("call", "put"):
                price = float(bsm_price(kind, spot, strike, t, 0.0, 0.0, iv))
                vol = float(rng.integers(0, 2000))
                rows.append(
                    {
                        "expiry": pd.Timestamp(today) + pd.Timedelta(days=days),
                        "kind": kind,
                        "strike": round(float(strike), 2),
                        "bid": round(price * 0.99, 4),
                        "ask": round(price * 1.01, 4),
                        "last": round(price, 4),
                        "volume": vol,
                        "open_interest": float(rng.integers(100, 20000)),
                        "provider_iv": iv,
                        "contract": f"{symbol}{days}{kind[0].upper()}{strike:.0f}",
                        "last_trade": pd.Timestamp(datetime.now(UTC)),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["expiry"] = frame["expiry"].dt.date
    return frame


# ── lake ─────────────────────────────────────────────────────────────────────


def _chain_dir(lake: PriceLake, symbol: str) -> Path:
    return lake.root / "option_chains" / f"symbol={symbol.upper()}"


def write_chain(
    lake: PriceLake, symbol: str, as_of: date, spot: float, frame: pd.DataFrame
) -> None:
    out = frame[CHAIN_COLUMNS].copy()
    out["spot"] = spot
    target = _chain_dir(lake, symbol) / f"as_of={as_of.isoformat()}" / "part.parquet"
    _atomic_write(out, pa.Table.from_pandas(out, preserve_index=False).schema, target)


def chain_dates(lake: PriceLake, symbol: str) -> list[date]:
    root = _chain_dir(lake, symbol)
    if not root.exists():
        return []
    return sorted(
        date.fromisoformat(p.name.split("=", 1)[1])
        for p in root.glob("as_of=*")
        if (p / "part.parquet").exists()
    )


def read_chain(
    lake: PriceLake, symbol: str, as_of: date | None = None
) -> tuple[date, float, pd.DataFrame] | None:
    dates = chain_dates(lake, symbol)
    if not dates:
        return None
    chosen = dates[-1] if as_of is None else max((d for d in dates if d <= as_of), default=None)
    if chosen is None:
        return None
    frame = pd.read_parquet(
        _chain_dir(lake, symbol) / f"as_of={chosen.isoformat()}" / "part.parquet"
    )
    return chosen, float(frame["spot"].iloc[0]), frame.drop(columns="spot")


def _iv_file(lake: PriceLake, symbol: str) -> Path:
    return lake.root / "option_iv" / f"symbol={symbol.upper()}" / "part.parquet"


def append_iv_history(lake: PriceLake, symbol: str, row: dict[str, Any]) -> None:
    target = _iv_file(lake, symbol)
    new = pd.DataFrame([row])
    if target.exists():
        new = pd.concat([pq.read_table(target).to_pandas(), new], ignore_index=True)
        new = new.drop_duplicates(subset=["date"], keep="last")
    _atomic_write(new.sort_values("date", ignore_index=True), IV_SCHEMA, target)


def read_iv_history(lake: PriceLake, symbol: str) -> pd.DataFrame:
    target = _iv_file(lake, symbol)
    if not target.exists():
        return pd.DataFrame(columns=IV_SCHEMA.names)
    return pq.read_table(target).to_pandas()


def snapshot_chain(
    lake: PriceLake, symbol: str, today: date, spot: float, frame: pd.DataFrame, r: float, q: float
) -> dict[str, Any]:
    """Stores the chain and its daily summary; returns the summary."""
    write_chain(lake, symbol, today, spot, frame)
    summary = chain_summary(enrich_chain(frame, spot, today, r, q), spot)
    row = {"date": today, "spot": spot, **summary}
    append_iv_history(lake, symbol, row)
    return row


# ── ingestion job ────────────────────────────────────────────────────────────

SHORT_RATE = {"USD": "DTB3", "EUR": "ESTR"}


def carry_inputs(
    lake: PriceLake, instrument: dict[str, Any], spot: float, today: date
) -> tuple[float, float]:
    """Short rate of the currency (decimal; 0 when not stored) and trailing 12-month dividend
    yield from stored actions (0 without dividends)."""
    r = 0.0
    code = SHORT_RATE.get(str(instrument.get("currency")))
    if code:
        series = lake.read_series(code)
        if not series.empty:
            r = float(series["value"].iloc[-1]) / 100
    actions = lake.read_actions(int(instrument["id"]))
    q = 0.0
    if not actions.empty and spot > 0:
        recent = actions.loc[
            (actions["action"] == "dividend")
            & (pd.to_datetime(actions["ex_date"]) > pd.Timestamp(today) - pd.Timedelta(days=365))
        ]
        q = float(recent["amount"].sum()) / spot
    return r, q


def ingest_options(
    lake: PriceLake, instrument: dict[str, Any], last_close: float | None, today: date
) -> dict[str, Any]:
    """Fetches today's chain (synthetic for FAKE* symbols), stores it and its IV summary."""
    symbol = str(instrument["symbol"]).upper()
    if symbol.startswith("FAKE"):
        if last_close is None:
            raise ProviderError(
                "Ingest the prices first: the synthetic chain is centred on the last close"
            )
        spot, frame = last_close, fake_chain(symbol, today, last_close)
    else:
        spot, frame = fetch_yahoo_chain(symbol, today)
    r, q = carry_inputs(lake, instrument, spot, today)
    summary = snapshot_chain(lake, symbol, today, spot, frame, r, q)
    return {
        "symbol": symbol,
        "contracts": len(frame),
        "expiries": int(frame["expiry"].nunique()),
        "r": r,
        "q": q,
        **summary,
    }
