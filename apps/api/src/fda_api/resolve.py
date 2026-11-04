"""Resolves an analysis' `requires` into engine inputs (TRD §3.2): data + type conventions.

Two phases keep warm runs fast: `resolve()` only reads catalog metadata to build the
`data_version` (part of the cache key, so new sessions, re-classification or overrides
invalidate cached results); the heavy frames (prices from the lake, PIT facts) are loaded by
`Resolved.inputs()` on a cache miss only, once per request thanks to the shared `memo`.
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import Engine, select

from fda_contracts.envelopes import RunRequest
from fda_core.analyses.invest import CLASSES, CMA_SERIES, classify_holding
from fda_core.analyses.markets import MARKET_INPUTS
from fda_core.asset_types import Classification, conventions
from fda_core.backtest import ordered_constraint, param_grid
from fda_core.portfolio import PortfolioInputs, build_path, fx_rate
from fda_core.registry import AnalysisInputs, AnalysisSpec, InsufficientDataError
from fda_data.company import load_peer_group, load_snapshots
from fda_data.filings import filings_version, read_filings
from fda_data.fundamentals import (
    FUNDAMENTALS_DATASET,
    load_all_facts,
    load_facts,
    mapping_version,
)
from fda_data.ingest import PRICES_DATASET
from fda_data.lake import PriceLake
from fda_data.news import news_version, read_news
from fda_data.options import carry_inputs, read_chain, read_iv_history
from fda_data.paper import cost_model, load_account, to_orders
from fda_data.portfolios import PortfolioError, load_portfolio
from fda_data.schema import comparisons, instruments, investor_profiles, series_catalog
from fda_data.strategies import load_strategy, record_trials
from fda_data.universes import UniverseError, load_universe, read_snapshot

SYNTHETIC_DATA_VERSION = "synthetic-v1"

Memo = dict[tuple[Any, ...], Any]

# Catalog rows change only when a job finishes: a 2 s process-wide TTL keeps warm panel
# refreshes off the database (each Postgres round-trip through Docker costs a few ms).
ROW_TTL_S = 2.0
_rows: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}


@dataclass(slots=True)
class Resolved:
    data_version: str
    conventions: dict[str, Any] = field(default_factory=dict)
    loaders: dict[str, Callable[[], Any]] = field(default_factory=dict)

    def inputs(self) -> AnalysisInputs:
        return AnalysisInputs(
            data={name: load() for name, load in self.loaders.items()},
            conventions=self.conventions,
        )


def _memoised(memo: Memo, key: tuple[Any, ...], load: Callable[[], Any]) -> Any:
    if key not in memo:
        memo[key] = load()
    return memo[key]


def _catalog_row(engine: Engine, symbol: str, dataset: str, memo: Memo) -> dict[str, Any] | None:
    def load() -> dict[str, Any] | None:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    select(
                        instruments,
                        series_catalog.c.last_date,
                        series_catalog.c.rows,
                        series_catalog.c.last_ingested_at,
                    )
                    .outerjoin(
                        series_catalog,
                        (series_catalog.c.instrument_id == instruments.c.id)
                        & (series_catalog.c.dataset == dataset),
                    )
                    .where(instruments.c.symbol == symbol.upper())
                )
                .mappings()
                .one_or_none()
            )
        return dict(row) if row is not None else None

    def cached() -> dict[str, Any] | None:
        key = (symbol.upper(), dataset)
        hit = _rows.get(key)
        if hit is not None and hit[0] > time.monotonic():
            return hit[1]
        row = load()
        if row is not None and row["rows"] is not None:  # never cache "not ingested yet"
            _rows[key] = (time.monotonic() + ROW_TTL_S, row)
        return row

    result: dict[str, Any] | None = _memoised(memo, ("row", symbol.upper(), dataset), cached)
    return result


def _instrument(engine: Engine, symbol: str, memo: Memo | None = None) -> dict[str, Any]:
    row = _catalog_row(engine, symbol, PRICES_DATASET, {} if memo is None else memo)
    if row is None or row["rows"] is None:
        raise InsufficientDataError(f"{symbol.upper()} has no stored prices: ingest it first")
    return row


def type_conventions(row: dict[str, Any]) -> dict[str, Any]:
    profile = row["profile"] or {}
    classification = Classification(
        row["asset_class"],
        row["asset_subclass"] or "",
        row["classification_source"] or "low",
        row["classification_reason"] or "",
        float(row["leverage_factor"]) if row["leverage_factor"] is not None else None,
        tuple(profile.get("flags") or ()),
    )
    conv = conventions(classification, row["currency"])
    return {
        "asset_class": classification.asset_class,
        "asset_subclass": classification.asset_subclass,
        "currency": row["currency"],
        "annualisation": conv.annualisation,
        "calendar": conv.calendar,
        "price_field": conv.price_field if conv.price_field != "level" else "close",
        "has_volume": conv.has_volume,
        "return_kind": conv.return_kind,
        "benchmark": conv.benchmark,
        "caveats": list(conv.caveats),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "country": row.get("country"),
        "leverage_factor": float(row["leverage_factor"])
        if row["leverage_factor"] is not None
        else None,
    }


def _frame(lake: PriceLake, row: dict[str, Any]) -> pd.DataFrame:
    frame = lake.read_prices(row["asset_class"], row["id"])
    return frame.assign(session_date=pd.to_datetime(frame["session_date"])).set_index(
        "session_date"
    )


def _prices_loader(lake: PriceLake, row: dict[str, Any], memo: Memo) -> Callable[[], Any]:
    return lambda: _memoised(
        memo, ("prices", row["id"], row["last_date"]), lambda: _frame(lake, row)
    )


def resolve(
    spec: AnalysisSpec,
    request: RunRequest,
    engine: Engine,
    lake: PriceLake,
    params: BaseModel | None = None,
    memo: Memo | None = None,
) -> Resolved:
    memo = {} if memo is None else memo
    if not spec.requires:
        return Resolved(SYNTHETIC_DATA_VERSION)
    if request.subject.type == "portfolio":
        return _resolve_portfolio(spec, request, engine, lake, params, memo)
    if request.subject.type == "comparison":
        return _resolve_comparison(request, engine, lake, memo)
    if request.subject.type == "market":
        return _resolve_market(spec, request, engine, lake, memo)
    if request.subject.type == "universe":
        return _resolve_universe(spec, request, engine, lake, memo)
    if request.subject.type == "profile":
        return _resolve_profile(spec, request, engine, lake, params, memo)
    if request.subject.type == "strategy":
        return _resolve_strategy(request, engine, lake, memo)
    if request.subject.type == "paper":
        return _resolve_paper(request, engine, lake, memo)
    if request.subject.type != "asset":
        raise InsufficientDataError(f"{spec.id} needs an asset subject")
    symbol = request.subject.id
    if spec.requires == {"filings"}:
        resolved = Resolved(
            f"filings:{symbol.upper()}:{filings_version(lake, symbol)}",
            {"range": request.context.range},
        )
        resolved.loaders["filings"] = lambda: read_filings(lake, symbol)
        return resolved
    if spec.requires & {"peers", "company"} and not spec.requires & {
        "fundamentals",
        "fundamentals_all",
        "prices",
    }:
        row = _catalog_row(engine, symbol, PRICES_DATASET, memo)
        if row is None:
            raise InsufficientDataError(f"{symbol.upper()} is unknown: ingest it first")
        resolved = Resolved("", type_conventions(row) | {"range": request.context.range})
        _attach_company(resolved, engine, int(row["id"]), spec.requires, memo)
        return resolved
    if spec.requires & {"fundamentals", "fundamentals_all"}:
        resolved = _resolve_fundamentals(engine, symbol, memo, spec.requires)
        _risk_free(resolved, lake, memo)
        if "other_prices" in spec.requires:
            _attach_other(resolved, engine, lake, params, memo, symbol)
        if spec.requires & {"peers", "company"}:
            row_f = _catalog_row(engine, symbol, FUNDAMENTALS_DATASET, memo)
            if row_f is not None:
                _attach_company(resolved, engine, int(row_f["id"]), spec.requires, memo)
        if "prices" in spec.requires:
            # Market-value inputs (market cap at fiscal year ends): prices are optional here.
            try:
                price_row = _instrument(engine, symbol, memo)
            except InsufficientDataError:
                return resolved
            resolved.loaders["prices"] = _prices_loader(lake, price_row, memo)
            resolved.data_version += f"|prices:{price_row['last_date']}"
            if "actions" in spec.requires:
                instrument_id = int(price_row["id"])
                resolved.loaders["actions"] = lambda: _memoised(
                    memo, ("actions", instrument_id), lambda: lake.read_actions(instrument_id)
                )
        return resolved
    row = _instrument(engine, symbol, memo)
    conv = type_conventions(row) | {"range": request.context.range}
    if conv["annualisation"] is None:
        conv["price_field"] = "close"
    resolved = Resolved(
        f"{row['last_date']}:{row['rows']}:{row['asset_class']}/{row['asset_subclass']}:"
        f"{row['leverage_factor']}",
        conv,
        {"prices": _prices_loader(lake, row, memo)},
    )
    if spec.requires & {"peers", "company"}:
        _attach_company(resolved, engine, int(row["id"]), spec.requires, memo)
    if "news" in spec.requires:
        resolved.data_version += f"|news:{news_version(lake, symbol)}"
        resolved.loaders["news"] = lambda: read_news(lake, symbol)
    if "other_prices" in spec.requires:
        # Second series: explicit `other` param, else the benchmark of the asset type.
        other = getattr(params, "other", None) or conv.get("benchmark")
        if other and str(other).upper() != row["symbol"]:
            conv["other_symbol"] = str(other).upper()
            try:
                other_row = _instrument(engine, str(other), memo)
            except InsufficientDataError:
                other_row = None  # analyses needing it explain how to fix it
            if other_row is not None:
                resolved.loaders["other_prices"] = _prices_loader(lake, other_row, memo)
                resolved.data_version += f"|{other_row['symbol']}:{other_row['last_date']}"
    _attach_quant(resolved, spec, engine, lake, params, memo, row["symbol"])
    if spec.requires & {"options", "option_history", "vol_proxy"}:
        _attach_options(resolved, spec, engine, lake, row, memo)
    return resolved


def _attach_company(
    resolved: Resolved, engine: Engine, instrument_id: int, requires: frozenset[str], memo: Memo
) -> None:
    """Peer group and provider snapshots: small, read eagerly to version the cache key."""
    if "peers" in requires:
        group = _memoised(
            memo, ("peers", instrument_id), lambda: load_peer_group(engine, instrument_id)
        )
        tag = f"{group['created_at']}:{group['mode']}:{len(group['members'])}" if group else "none"
        resolved.data_version += f"|peers:{tag}"
        if group is not None:
            resolved.loaders["peers"] = lambda: group
    if "company" in requires:
        snaps = _memoised(
            memo, ("company", instrument_id), lambda: load_snapshots(engine, instrument_id)
        )
        resolved.data_version += (
            f"|company:{max((v['as_of'] for v in snaps.values()), default=None)}"
        )
        if snaps:
            resolved.loaders["company"] = lambda: snaps


RISK_FREE_SERIES = {"USD": "DTB3", "EUR": "ESTR"}


def _risk_free(resolved: Resolved, lake: PriceLake, memo: Memo) -> None:
    """Latest stored short rate for the currency (DTB3 / €STR), as a decimal; else unset."""
    code = RISK_FREE_SERIES.get(str(resolved.conventions.get("currency")))
    if code is None:
        return
    series = _memoised(memo, ("series", code), lambda: lake.read_series(code))
    if series.empty:
        return
    resolved.conventions["rf"] = float(series["value"].iloc[-1]) / 100
    last = pd.Timestamp(series["date"].iloc[-1]).date().isoformat()
    resolved.conventions["rf_source"] = f"{code} {last}"
    resolved.data_version += f"|rf:{code}:{last}"


def _attach_other(
    resolved: Resolved,
    engine: Engine,
    lake: PriceLake,
    params: BaseModel | None,
    memo: Memo,
    symbol: str,
) -> None:
    other = getattr(params, "other", None) or resolved.conventions.get("benchmark")
    if not other or str(other).upper() == symbol.upper():
        return
    resolved.conventions["other_symbol"] = str(other).upper()
    try:
        other_row = _instrument(engine, str(other), memo)
    except InsufficientDataError:
        return
    resolved.loaders["other_prices"] = _prices_loader(lake, other_row, memo)
    resolved.data_version += f"|{other_row['symbol']}:{other_row['last_date']}"


def _resolve_fundamentals(
    engine: Engine, symbol: str, memo: Memo, requires: frozenset[str] = frozenset({"fundamentals"})
) -> Resolved:
    """Point-in-time facts (latest filing known today) + the instrument's type conventions."""
    row = _catalog_row(engine, symbol, FUNDAMENTALS_DATASET, memo)
    if row is None:
        raise InsufficientDataError(f"{symbol.upper()} is unknown: ingest it first")
    conv = type_conventions(row)
    if conv["asset_class"] == "equity" and row["rows"] is None:
        raise InsufficientDataError(
            f"No statements stored for {symbol.upper()}: ingest fundamentals"
        )
    data_version = (
        f"fundamentals:{row['last_ingested_at']}:{row['rows']}:map{mapping_version()}:"
        f"{row['asset_class']}/{row['asset_subclass']}"
    )
    loaders: dict[str, Callable[[], Any]] = {}
    if row["rows"] is not None and "fundamentals_all" in requires:
        all_id = int(row["id"])
        loaders["fundamentals_all"] = lambda: _memoised(
            memo,
            ("facts_all", all_id, row["last_ingested_at"]),
            lambda: load_all_facts(engine, all_id),
        )
    if row["rows"] is not None and "fundamentals" in requires:
        instrument_id = int(row["id"])
        loaders["fundamentals"] = lambda: _memoised(
            memo,
            ("facts", instrument_id, row["last_ingested_at"]),
            lambda: load_facts(engine, instrument_id),
        )
    return Resolved(data_version, conv, loaders)


# ── Portfolios (M15): the Subject abstraction ────────────────────────────────

PORTFOLIO_ID = re.compile(r"^(\d+)(?:@(\d+))?$")  # "12" (current version) or "12@3"
FACTOR_PROXIES = {
    "equity": ("SPY", "^GSPC"),
    "rates_10y": ("^TNX",),
    "usd": ("DX-Y.NYB", "UUP"),
    "oil": ("CL=F", "USO"),
}
FACTOR_SERIES = {"credit": "BAMLH0A0HYM2"}
MACRO_SERIES = FACTOR_SERIES | {"inflation": "T5YIE"}  # Q-011 adds the 5y breakeven
KF_FACTORS = ("MKTRF", "SMB", "HML", "RMW", "CMA", "RF", "MOM")


def _definition(engine: Engine, pid: int, version: int | None, memo: Memo) -> dict[str, Any]:
    def load() -> dict[str, Any]:
        try:
            return load_portfolio(engine, pid, version)
        except PortfolioError as exc:
            raise InsufficientDataError(str(exc)) from exc

    result: dict[str, Any] = _memoised(memo, ("pf-def", pid, version), load)
    return result


def _fx_row(engine: Engine, ccy: str, base: str, memo: Memo) -> tuple[dict[str, Any], bool] | None:
    """Stored FX pair giving base per unit of ccy: CCYBASE=X, else BASECCY=X inverted."""
    for symbol, invert in ((f"{ccy}{base}=X", False), (f"{base}{ccy}=X", True)):
        row = _catalog_row(engine, symbol, PRICES_DATASET, memo)
        if row is not None and row["rows"] is not None:
            return row, invert
    return None


PortfolioParts = tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, tuple[dict[str, Any], bool]], str
]


def _portfolio_parts(engine: Engine, pid: int, version: int | None, memo: Memo) -> PortfolioParts:
    """Definition, instrument rows, FX rows and the data version: metadata only."""
    d = _definition(engine, pid, version, memo)
    if d["mode"] == "weights":
        symbols = [h["symbol"] for h in d["holdings"] if not h["is_cash"]]
    else:
        symbols = list(d["symbols"])
    rows: dict[str, dict[str, Any]] = {}
    missing = []
    for symbol in symbols:
        row = _catalog_row(engine, symbol, PRICES_DATASET, memo)
        if row is None or row["rows"] is None:
            missing.append(symbol)
        else:
            rows[symbol] = row
    if missing:
        raise InsufficientDataError(f"Ingest the holdings first: {', '.join(missing)}")
    if not rows:
        raise InsufficientDataError("The portfolio has no holdings yet")
    base = d["base_currency"]
    currencies = {str(r["currency"]) for r in rows.values()}
    if d["mode"] == "transactions" and d["transactions"] is not None:
        currencies |= {str(c) for c in d["transactions"]["currency"].dropna()}
    fx: dict[str, tuple[dict[str, Any], bool]] = {}
    for ccy in sorted(currencies - {base}):
        found = _fx_row(engine, ccy, base, memo)
        if found is None:
            raise InsufficientDataError(f"Ingest {ccy}{base}=X to convert {ccy} holdings to {base}")
        fx[ccy] = found
    tag = ",".join(f"{s}:{r['last_date']}" for s, r in sorted(rows.items()))
    fx_tag = ",".join(f"{c}:{r['last_date']}" for c, (r, _) in sorted(fx.items()))
    return d, rows, fx, f"pf:{d['hash']}:{d['ledger_tag']}:{tag}|fx:{fx_tag}"


def _portfolio_inputs(
    engine: Engine, lake: PriceLake, pid: int, version: int | None, memo: Memo
) -> PortfolioInputs:
    d, rows, fx_rows, data_version = _portfolio_parts(engine, pid, version, memo)

    def frame_of(row: dict[str, Any]) -> pd.DataFrame:
        result: pd.DataFrame = _memoised(
            memo, ("prices", row["id"], row["last_date"]), lambda: _frame(lake, row)
        )
        return result

    def build() -> PortfolioInputs:
        frames = {s: frame_of(r) for s, r in rows.items()}
        fx: dict[str, pd.Series] = {}
        for ccy, (row, invert) in fx_rows.items():
            close = frame_of(row)["close"]
            fx[ccy] = 1 / close if invert else close
        actions = {s: lake.read_actions(int(r["id"])) for s, r in rows.items()}
        currencies = {s: str(r["currency"]) for s, r in rows.items()}
        keys = (
            "id",
            "name",
            "mode",
            "base_currency",
            "benchmark",
            "holdings",
            "rebalance",
            "cost_bps",
            "start_date",
            "initial_value",
            "lot_method",
        )
        definition = {k: d[k] for k in keys}
        if d["mode"] == "transactions":
            definition["transactions"] = d["transactions"]
        seven_day = {s for s, r in rows.items() if r["asset_class"] == "crypto"}
        path = build_path(definition, frames, currencies, fx, actions, seven_day)
        meta = {}
        for s, r in rows.items():
            snaps = load_snapshots(engine, int(r["id"]))
            info = (snaps.get("info") or {}).get("payload") or {}
            meta[s] = type_conventions(r) | {"name": r.get("name"), "info": info}
        fx_last = {c: float(series.dropna().iloc[-1]) for c, series in fx.items()}
        fx_last[d["base_currency"]] = 1.0
        return PortfolioInputs(path, definition, meta, frames, actions, fx_last, fx)

    inputs: PortfolioInputs = _memoised(memo, ("pf", pid, version, data_version), build)
    return inputs


def _nav_frame(pf: PortfolioInputs) -> pd.DataFrame:
    nav = pf.path.nav
    columns: dict[str, Any] = {c: nav for c in ("open", "high", "low", "close", "adj_close")}
    return pd.DataFrame(columns | {"volume": float("nan")})


def _resolve_portfolio(
    spec: AnalysisSpec,
    request: RunRequest,
    engine: Engine,
    lake: PriceLake,
    params: BaseModel | None,
    memo: Memo,
) -> Resolved:
    match = PORTFOLIO_ID.match(request.subject.id)
    if match is None:
        raise InsufficientDataError(f"Unknown portfolio id {request.subject.id!r}")
    pid, version = int(match[1]), int(match[2]) if match[2] else None
    d, rows, _, data_version = _portfolio_parts(engine, pid, version, memo)
    classes = {r["asset_class"] for r in rows.values()}
    base = d["base_currency"]
    benchmark = str(d["benchmark"] or "")
    bench = benchmark if benchmark and not benchmark.startswith("portfolio:") else None
    mixed = "crypto" in classes and len(classes) > 1
    conv: dict[str, Any] = {
        "asset_class": "portfolio",
        "asset_subclass": "portfolio",
        "currency": base,
        "annualisation": 365 if classes == {"crypto"} else 252,
        "calendar": "mixed" if mixed else None,
        "price_field": "adj_close",
        "has_volume": False,
        "return_kind": "simple",
        "benchmark": bench or ("SPY" if base == "USD" else "^STOXX"),
        "range": request.context.range,
        "caveats": ["Crypto holdings are sampled on exchange sessions (mixed calendars)"]
        if mixed
        else [],
    }
    resolved = Resolved(data_version, conv)
    resolved.loaders["portfolio"] = lambda: _portfolio_inputs(engine, lake, pid, version, memo)
    resolved.loaders["prices"] = lambda: _nav_frame(
        _portfolio_inputs(engine, lake, pid, version, memo)
    )
    _risk_free(resolved, lake, memo)
    if "other_prices" in spec.requires:
        _attach_other(resolved, engine, lake, params, memo, f"portfolio:{pid}")
    if "compare_portfolio" in spec.requires:
        other = getattr(params, "compare", None)
        if other is None and benchmark.startswith("portfolio:"):
            other = int(benchmark.split(":", 1)[1])
        if other is not None:
            other_id = int(other)
            other_version = _portfolio_parts(engine, other_id, None, memo)[3]
            resolved.data_version += f"|vs:{other_version}"
            resolved.loaders["compare_portfolio"] = lambda: _portfolio_inputs(
                engine, lake, other_id, None, memo
            )
    _attach_quant(resolved, spec, engine, lake, params, memo, f"portfolio:{pid}")
    return resolved


def _attach_quant(
    resolved: Resolved,
    spec: AnalysisSpec,
    engine: Engine,
    lake: PriceLake,
    params: BaseModel | None,
    memo: Memo,
    subject: str,
) -> None:
    """Inputs of factor and quant analyses (P-023, M18), for asset and portfolio subjects."""
    if "factors" in spec.requires:
        _attach_factors(resolved, engine, lake, memo)
    if "macro" in spec.requires:
        _attach_factors(resolved, engine, lake, memo, "macro", MACRO_SERIES)
    if "ff_factors" in spec.requires:
        _attach_ken_french(resolved, lake, memo)
    if "regressors" in spec.requires:
        _attach_regressors(resolved, engine, lake, params, memo, subject)
    if "vix" in spec.requires:
        row = _catalog_row(engine, "^VIX", PRICES_DATASET, memo)
        if row is not None and row["rows"] is not None:
            resolved.conventions["vix_symbol"] = "^VIX"
            resolved.data_version += f"|vix:{row['last_date']}:{row['rows']}"
            loader = _prices_loader(lake, row, memo)
            resolved.loaders["vix"] = lambda: loader()["close"].astype(float)
        else:
            resolved.data_version += "|vix:none"


def _attach_ken_french(resolved: Resolved, lake: PriceLake, memo: Memo) -> None:
    """Daily Ken French factors per region (KF_<REGION>_<FACTOR> series), as decimals."""
    stored: dict[str, dict[str, pd.DataFrame]] = {}
    for region in ("US", "EU"):
        frames = {}
        for factor in KF_FACTORS:
            code = f"KF_{region}_{factor}"
            frame = _memoised(memo, ("series", code), lambda code=code: lake.read_series(code))
            if not frame.empty:
                frames[factor] = frame
        if frames:
            stored[region] = frames
    tags = [
        f"{region}:{max(pd.Timestamp(fr['date'].iloc[-1]) for fr in frames.values()).date()}"
        for region, frames in sorted(stored.items())
    ]
    resolved.data_version += "|kf:" + (",".join(tags) or "none")

    def load() -> dict[str, pd.DataFrame]:
        out = {}
        for region, frames in stored.items():
            columns = {
                f: fr.assign(date=pd.to_datetime(fr["date"])).set_index("date")["value"]
                for f, fr in frames.items()
            }
            out[region] = pd.DataFrame(columns).astype(float)
        return out

    resolved.loaders["ff_factors"] = load


def _attach_regressors(
    resolved: Resolved,
    engine: Engine,
    lake: PriceLake,
    params: BaseModel | None,
    memo: Memo,
    subject: str,
) -> None:
    """Q-001 X series: the `regressors` param (≤ 8 symbols) or the type's benchmark."""
    text = str(getattr(params, "regressors", "") or resolved.conventions.get("benchmark") or "")
    symbols = [s.strip().upper() for s in text.split(",") if s.strip()][:8]
    rows = {}
    for symbol in symbols:
        if symbol == subject.upper():
            continue
        try:
            rows[symbol] = _instrument(engine, symbol, memo)
        except InsufficientDataError:
            continue
    resolved.conventions["regressors"] = list(rows)
    resolved.data_version += "|x:" + ",".join(f"{s}:{r['last_date']}" for s, r in rows.items())
    resolved.loaders["regressors"] = lambda: {
        s: _prices_loader(lake, r, memo)()[str(type_conventions(r)["price_field"])]
        for s, r in rows.items()
    }


def _attach_factors(
    resolved: Resolved,
    engine: Engine,
    lake: PriceLake,
    memo: Memo,
    key: str = "factors",
    fred: dict[str, str] = FACTOR_SERIES,
) -> None:
    """Stored factor proxies for P-023 (first stored symbol per factor, FRED series)."""
    rows: dict[str, dict[str, Any]] = {}
    for name, symbols in FACTOR_PROXIES.items():
        for symbol in symbols:
            row = _catalog_row(engine, symbol, PRICES_DATASET, memo)
            if row is not None and row["rows"] is not None:
                rows[name] = row
                break
    series: dict[str, pd.DataFrame] = {}
    for name, code in fred.items():
        frame = lake.read_series(code)
        if not frame.empty:
            series[name] = frame
    tags = [f"{n}:{r['last_date']}" for n, r in sorted(rows.items())]
    tags += [f"{n}:{len(f)}" for n, f in sorted(series.items())]
    resolved.data_version += f"|{key}:" + ",".join(tags)

    def load() -> dict[str, pd.Series]:
        out = {n: _frame(lake, r)["adj_close"] for n, r in rows.items()}
        for n, f in series.items():
            out[n] = (
                f.assign(date=pd.to_datetime(f["date"])).set_index("date")["value"].astype(float)
            )
        return out

    resolved.loaders[key] = load


# ── Comparisons (M16): members as level series in one currency ───────────────

MAX_MEMBERS = 8


def parse_members(text: str) -> list[dict[str, str]]:
    """ "AAPL, portfolio:3, asset:SPY, portfolio:3@2" → subject refs (max 8)."""
    refs = []
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        kind, _, ref = token.partition(":") if ":" in token else ("asset", "", token)
        if kind not in ("asset", "portfolio") or not ref:
            raise InsufficientDataError(f"Unknown comparison member {token!r}")
        refs.append({"type": kind, "id": ref.upper() if kind == "asset" else ref})
    if not 2 <= len(refs) <= MAX_MEMBERS:
        raise InsufficientDataError(f"A comparison needs 2–{MAX_MEMBERS} members, got {len(refs)}")
    return refs


def _comparison_members(engine: Engine, subject_id: str) -> list[dict[str, str]]:
    if subject_id.isdigit():
        with engine.connect() as conn:
            row = conn.execute(
                select(comparisons.c.members).where(comparisons.c.id == int(subject_id))
            ).one_or_none()
        if row is None:
            raise InsufficientDataError(f"Comparison {subject_id} not found")
        return [dict(m) for m in row[0]]
    return parse_members(subject_id)


def _resolve_comparison(
    request: RunRequest, engine: Engine, lake: PriceLake, memo: Memo
) -> Resolved:
    refs = _comparison_members(engine, request.subject.id)
    parts: list[dict[str, Any]] = []
    for ref in refs:
        if ref["type"] == "asset":
            row = _instrument(engine, ref["id"], memo)
            conv = type_conventions(row)
            parts.append(
                {
                    "ref": ref,
                    "label": row["symbol"],
                    "row": row,
                    "currency": row["currency"],
                    "annualisation": conv["annualisation"] or 252,
                    "field": conv["price_field"],
                    "version": f"{row['symbol']}:{row['last_date']}",
                    "kind": "asset",
                    "is_benchmark": False,
                }
            )
        else:
            match = PORTFOLIO_ID.match(ref["id"])
            if match is None:
                raise InsufficientDataError(f"Unknown portfolio id {ref['id']!r}")
            pid, version = int(match[1]), int(match[2]) if match[2] else None
            d, rows, _, tag = _portfolio_parts(engine, pid, version, memo)
            classes = {r["asset_class"] for r in rows.values()}
            parts.append(
                {
                    "ref": ref,
                    "label": d["name"] + (f" @v{version}" if version else ""),
                    "pid": pid,
                    "pversion": version,
                    "currency": d["base_currency"],
                    "annualisation": 365 if classes == {"crypto"} else 252,
                    "version": tag,
                    "kind": "portfolio",
                    "is_benchmark": bool(d["is_benchmark"]),
                }
            )
    ccy_param = request.context.ccy.upper()
    currency = ccy_param if len(ccy_param) == 3 and ccy_param != "LISTING" else parts[0]["currency"]
    fx_rows: dict[str, tuple[dict[str, Any], bool]] = {}
    for ccy in sorted({p["currency"] for p in parts} - {currency}):
        found = _fx_row(engine, ccy, currency, memo)
        if found is None:
            raise InsufficientDataError(f"Ingest {ccy}{currency}=X to compare in {currency}")
        fx_rows[ccy] = found
    seen: dict[str, int] = {}
    for part in parts:  # unique labels
        seen[part["label"]] = seen.get(part["label"], 0) + 1
        if seen[part["label"]] > 1:
            part["label"] += f" ({seen[part['label']]})"
    fx_tag = ",".join(f"{c}:{r['last_date']}" for c, (r, _) in sorted(fx_rows.items()))
    data_version = "cmp:" + "|".join(p["version"] for p in parts) + f"|{currency}|fx:{fx_tag}"
    conv = {"range": request.context.range, "currency": currency}
    resolved = Resolved(data_version, conv)

    def load() -> list[dict[str, Any]]:
        fx = {}
        for ccy, (row, invert) in fx_rows.items():
            close = _memoised(
                memo, ("prices", row["id"], row["last_date"]), lambda row=row: _frame(lake, row)
            )["close"]
            fx[ccy] = 1 / close if invert else close
        members = []
        for part in parts:
            if part["kind"] == "asset":
                row = part["row"]
                frame = _memoised(
                    memo, ("prices", row["id"], row["last_date"]), lambda row=row: _frame(lake, row)
                )
                series = frame[part["field"]].astype(float)
            else:
                series = _portfolio_inputs(
                    engine, lake, part["pid"], part["pversion"], memo
                ).path.nav
            if part["currency"] != currency:
                series = series * fx_rate(fx, part["currency"], currency, series.index)
            members.append(
                {
                    "label": part["label"],
                    "series": series.dropna(),
                    "kind": part["kind"],
                    "annualisation": part["annualisation"],
                    "is_benchmark": part["is_benchmark"],
                    "ref": part["ref"],
                }
            )
        return members

    resolved.loaders["members"] = load
    _risk_free(resolved, lake, memo)
    return resolved


# ── Markets (M19): the instruments and series an MK analysis declares ────────


def _resolve_market(
    spec: AnalysisSpec, request: RunRequest, engine: Engine, lake: PriceLake, memo: Memo
) -> Resolved:
    """Stored inputs only; the version tags every input's last date, so a new close or a new
    macro release invalidates the cached result."""
    symbols, codes = MARKET_INPUTS.get(spec.id, ([], []))
    rows = {}
    for symbol in symbols:
        row = _catalog_row(engine, symbol, PRICES_DATASET, memo)
        if row is not None and row["rows"] is not None:
            rows[symbol] = row
    series = {}
    for code in codes:
        frame = _memoised(memo, ("series", code), lambda code=code: lake.read_series(code))
        if not frame.empty:
            series[code] = frame
    # Last date and row count: a backfill (longer history) changes the result too.
    tags = [f"{s}:{r['last_date']}:{r['rows']}" for s, r in sorted(rows.items())]
    tags += [
        f"{c}:{pd.Timestamp(f['date'].iloc[-1]).date()}:{len(f)}" for c, f in sorted(series.items())
    ]
    resolved = Resolved("market:" + ",".join(tags), {"range": request.context.range})

    def prices() -> dict[str, pd.Series]:
        out = {}
        for symbol, row in rows.items():
            field_name = str(type_conventions(row)["price_field"])
            out[symbol] = _prices_loader(lake, row, memo)()[field_name].astype(float)
        return out

    def levels() -> dict[str, pd.Series]:
        return {
            c: f.assign(date=pd.to_datetime(f["date"])).set_index("date")["value"].astype(float)
            for c, f in series.items()
        }

    resolved.loaders["market_prices"] = prices
    resolved.loaders["market_series"] = levels
    if "cma" in spec.requires:
        _attach_cma(resolved, engine, lake, memo)
    return resolved


# ── Universes (M20): latest metric snapshot, member prices for breadth ───────


def _resolve_universe(
    spec: AnalysisSpec, request: RunRequest, engine: Engine, lake: PriceLake, memo: Memo
) -> Resolved:
    try:
        uid = int(request.subject.id)
        universe = load_universe(engine, uid)
    except (ValueError, UniverseError) as exc:
        raise InsufficientDataError(f"Unknown universe {request.subject.id!r}") from exc
    resolved = Resolved(f"universe:{uid}", {"range": request.context.range})
    if "snapshot" in spec.requires:
        snap = read_snapshot(lake, uid)
        if snap is not None:
            as_of, frame = snap
            resolved.conventions["as_of"] = as_of.isoformat()
            # Content hash: a same-day rebuild with new data must not hit the old cache entry.
            digest = int(pd.util.hash_pandas_object(frame, index=False).sum()) & 0xFFFFFFFF
            resolved.data_version += f"|snap:{as_of}:{len(frame)}:{digest:08x}"
            resolved.loaders["snapshot"] = lambda: frame
        else:
            resolved.data_version += "|snap:none"
    if "member_prices" in spec.requires:
        rows = {}
        for m in universe["members"]:
            row = _catalog_row(engine, m["symbol"], PRICES_DATASET, memo)
            if row is not None and row["rows"] is not None:
                rows[m["symbol"]] = row
        resolved.data_version += "|px:" + ",".join(
            f"{s}:{r['last_date']}:{r['rows']}" for s, r in sorted(rows.items())
        )
        resolved.loaders["member_prices"] = lambda: {
            s: _prices_loader(lake, r, memo)()["adj_close"].astype(float) for s, r in rows.items()
        }
    return resolved


# ── Options (M21): latest stored chain, IV history, VIX-family proxy, carry ──

VOL_PROXIES = {
    "SPY": "^VIX",
    "IVV": "^VIX",
    "VOO": "^VIX",
    "^GSPC": "^VIX",
    "^SPX": "^VIX",
    "QQQ": "^VXN",
    "^NDX": "^VXN",
    "IWM": "^RVX",
    "^RUT": "^RVX",
    "DIA": "^VXD",
}


def _attach_options(
    resolved: Resolved,
    spec: AnalysisSpec,
    engine: Engine,
    lake: PriceLake,
    row: dict[str, Any],
    memo: Memo,
) -> None:
    symbol = str(row["symbol"]).upper()
    _risk_free(resolved, lake, memo)
    resolved.conventions["rf"] = resolved.conventions.get("rf") or 0.0
    if "options" in spec.requires:
        snap = _memoised(memo, ("chain", symbol), lambda: read_chain(lake, symbol))
        if snap is not None:
            as_of, spot, chain = snap
            _, q = carry_inputs(lake, row, spot, as_of)
            resolved.conventions["dividend_yield"] = q
            resolved.data_version += f"|chain:{as_of}:{len(chain)}"
            shared = _memoised(
                memo,
                ("chain_inputs", symbol, as_of),
                lambda: {"as_of": as_of, "spot": spot, "chain": chain},
            )
            resolved.loaders["options"] = lambda: shared
        else:
            resolved.data_version += "|chain:none"
    if "option_history" in spec.requires:
        hist = read_iv_history(lake, symbol)
        resolved.data_version += f"|ivh:{len(hist)}"
        resolved.loaders["option_history"] = lambda: hist
    if "vol_proxy" in spec.requires and symbol in VOL_PROXIES:
        proxy = _catalog_row(engine, VOL_PROXIES[symbol], PRICES_DATASET, memo)
        if proxy is not None and proxy["rows"] is not None:
            resolved.conventions["vol_proxy_symbol"] = proxy["symbol"]
            resolved.data_version += f"|proxy:{proxy['symbol']}:{proxy['last_date']}"
            resolved.loaders["vol_proxy"] = lambda: _prices_loader(lake, proxy, memo)()[
                "close"
            ].astype(float)


# ── Investor profiles (M22): answers, capital-market inputs, current weights ─


def _attach_cma(resolved: Resolved, engine: Engine, lake: PriceLake, memo: Memo) -> None:
    """Long-history USD proxies per asset class, rates/spreads/breakeven, trailing yields."""
    rows = {}
    for _, proxy, _ in CLASSES.values():
        if proxy:
            row = _catalog_row(engine, proxy, PRICES_DATASET, memo)
            if row is not None and row["rows"] is not None:
                rows[proxy] = row
    series = {}
    for code in CMA_SERIES:
        frame = _memoised(memo, ("series", code), lambda code=code: lake.read_series(code))
        if not frame.empty:
            series[code] = frame
    tags = [f"{s}:{r['last_date']}:{r['rows']}" for s, r in sorted(rows.items())]
    tags += [f"{c}:{pd.Timestamp(f['date'].iloc[-1]).date()}" for c, f in sorted(series.items())]
    resolved.data_version += "|cma:" + ",".join(tags)

    def prices() -> dict[str, pd.Series]:
        return {
            s: _prices_loader(lake, r, memo)()["adj_close"].astype(float) for s, r in rows.items()
        }

    def yields() -> dict[str, float]:
        out = {}
        for symbol, row in rows.items():
            actions = lake.read_actions(int(row["id"]))
            close = _prices_loader(lake, row, memo)()["close"]
            if actions.empty or close.empty:
                continue
            cutoff = close.index[-1] - pd.Timedelta(days=365)
            div = actions.loc[
                (actions["action"] == "dividend") & (pd.to_datetime(actions["ex_date"]) > cutoff)
            ]
            out[symbol] = float(div["amount"].sum()) / float(close.iloc[-1])
        return out

    resolved.loaders["cma_prices"] = prices
    resolved.loaders["cma_series"] = lambda: {
        c: f.assign(date=pd.to_datetime(f["date"])).set_index("date")["value"].astype(float)
        for c, f in series.items()
    }
    resolved.loaders["cma_yields"] = yields


def load_profile(engine: Engine, profile_id: int) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(investor_profiles).where(investor_profiles.c.id == profile_id))
            .mappings()
            .one_or_none()
        )
    return dict(row) if row is not None else None


def _resolve_profile(
    spec: AnalysisSpec,
    request: RunRequest,
    engine: Engine,
    lake: PriceLake,
    params: BaseModel | None,
    memo: Memo,
) -> Resolved:
    try:
        profile = load_profile(engine, int(request.subject.id))
    except ValueError:
        profile = None
    if profile is None:
        raise InsufficientDataError(f"Unknown investor profile {request.subject.id!r}")
    resolved = Resolved(
        f"profile:{profile['id']}:{profile['scoring_version']}", {"range": request.context.range}
    )
    resolved.loaders["profile"] = lambda: {"name": profile["name"], "answers": profile["answers"]}
    if "cma" in spec.requires:
        _attach_cma(resolved, engine, lake, memo)
    compare = getattr(params, "compare", None)
    if compare is not None:
        version = _portfolio_parts(engine, int(compare), None, memo)[3]
        resolved.data_version += f"|vs:{version}"

        def current() -> dict[str, float]:
            pf = _portfolio_inputs(engine, lake, int(compare), None, memo)
            weights = pf.end_weights()
            out: dict[str, float] = {}
            for symbol, w in weights.items():
                meta = pf.meta.get(str(symbol), {})
                key = classify_holding(str(symbol), meta.get("asset_class"), meta.get("country"))
                out[key] = out.get(key, 0.0) + float(w)
            return out

        resolved.loaders["current_weights"] = current
    return resolved


# ── Strategies (M25): spec + prices; every distinct grid run enters the trial ledger ──


def _resolve_strategy(request: RunRequest, engine: Engine, lake: PriceLake, memo: Memo) -> Resolved:
    try:
        strat = load_strategy(engine, int(request.subject.id))
    except ValueError:
        strat = None
    if strat is None:
        raise InsufficientDataError(f"Unknown strategy {request.subject.id!r}")
    spec = strat["spec"]
    row = _instrument(engine, spec["symbol"], memo)
    grid = param_grid(
        {k: [float(x) for x in v] for k, v in spec["grid"].items()},
        ordered_constraint(spec["template"]),
    )
    total = _memoised(
        memo,
        ("trials", strat["id"], strat["updated_at"]),
        lambda: record_trials(engine, int(strat["id"]), {"spec": spec}, len(grid)),
    )
    resolved = Resolved(
        f"strategy:{strat['id']}:{strat['updated_at']}:{row['last_date']}:{row['rows']}:trials={total}",
        type_conventions(row) | {"range": request.context.range},
    )
    resolved.loaders["strategy"] = lambda: strat | {"trials": total}
    resolved.loaders["prices"] = _prices_loader(lake, row, memo)
    return resolved


# ── Paper accounts (M29): account + orders + the bars of every traded symbol ──


def _resolve_paper(request: RunRequest, engine: Engine, lake: PriceLake, memo: Memo) -> Resolved:
    try:
        account = load_account(engine, int(request.subject.id))
    except ValueError:
        account = None
    if account is None:
        raise InsufficientDataError(f"Unknown paper account {request.subject.id!r}")
    rows = {
        s: _instrument(engine, s, memo) for s in sorted({o["symbol"] for o in account["orders"]})
    }
    orders_key = hashlib.sha256(
        json.dumps(account["orders"], sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    bars_key = ",".join(f"{s}:{r['last_date']}:{r['rows']}" for s, r in rows.items())
    resolved = Resolved(
        f"paper:{account['id']}:{orders_key}:{bars_key}",
        {"range": request.context.range, "currency": account["currency"]},
    )

    def load() -> dict[str, Any]:
        return {
            "account": {
                k: account[k] for k in ("id", "name", "currency", "starting_cash", "costs")
            },
            "orders": to_orders(account["orders"]),
            "order_rows": account["orders"],
            "costs": cost_model(account["costs"]),
            "bars": {s: _frame(lake, r) for s, r in rows.items()},
        }

    resolved.loaders["paper"] = lambda: _memoised(memo, ("paper", resolved.data_version), load)
    return resolved
