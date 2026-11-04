"""Portfolio storage (BACKEND_SCHEMA §6.2): definitions as immutable versions, append-only ledger,
CSV import with validation, dry-run and idempotency (M15 builder)."""

import csv
import hashlib
import io
import json
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from fda_core.portfolio import TX_TYPES
from fda_data.schema import (
    import_batches,
    portfolio_targets,
    portfolio_versions,
    portfolios,
    transactions,
)

MODES = ("weights", "transactions")
LOT_METHODS = ("fifo", "lifo", "average")
CSV_COLUMNS = (
    "date",
    "type",
    "symbol",
    "quantity",
    "price",
    "currency",
    "fees",
    "fx_rate",
    "account",
    "external_id",
)
TRADE_TYPES = {"BUY", "SELL", "TRANSFER_IN", "TRANSFER_OUT", "DIVIDEND", "SPLIT_ADJ"}


class PortfolioError(ValueError):
    """Invalid definition or request (shown to the user)."""


def definition_hash(definition: dict[str, Any]) -> str:
    canonical = json.dumps(definition, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate(spec: dict[str, Any]) -> None:
    if spec.get("mode", "weights") not in MODES:
        raise PortfolioError(f"mode must be one of {MODES}")
    if spec.get("lot_method", "fifo") not in LOT_METHODS:
        raise PortfolioError(f"lot_method must be one of {LOT_METHODS}")
    if spec.get("mode", "weights") == "weights":
        holdings = spec.get("holdings") or []
        if not holdings:
            raise PortfolioError("Add at least one holding")
        total = sum(float(h["weight"]) for h in holdings)
        if abs(total - 1) > 1e-6:
            raise PortfolioError(f"Weights sum to {total:.6f}, not 1")
        symbols = [str(h["symbol"]).upper() for h in holdings]
        if len(set(symbols)) != len(symbols):
            raise PortfolioError("Duplicate holdings")


def _version_body(spec: dict[str, Any]) -> dict[str, Any]:
    holdings = [
        {
            "symbol": "CASH" if h.get("is_cash") else str(h["symbol"]).upper(),
            "weight": float(h["weight"]),
            "is_cash": bool(h.get("is_cash")),
            "locked": bool(h.get("locked")),
            "note": h.get("note"),
        }
        for h in spec.get("holdings") or []
    ]
    return {
        "mode": spec.get("mode", "weights"),
        "base_currency": str(spec.get("base_currency") or "USD").upper(),
        "benchmark": spec.get("benchmark"),
        "lot_method": spec.get("lot_method", "fifo"),
        "holdings": holdings,
        "rebalance": spec.get("rebalance") or {"type": "calendar", "freq": "M"},
        "cost_bps": float(spec.get("cost_bps") or 0.0),
        "start_date": str(spec["start_date"]) if spec.get("start_date") else None,
        "initial_value": float(spec.get("initial_value") or 100_000.0),
    }


def create_portfolio(engine: Engine, spec: dict[str, Any]) -> int:
    _validate(spec)
    body = _version_body(spec)
    name = str(spec.get("name") or "").strip()
    if not name:
        raise PortfolioError("Name the portfolio")
    with engine.begin() as conn:
        try:
            pid = int(
                conn.execute(
                    insert(portfolios).values(
                        name=name,
                        mode=body["mode"],
                        base_currency=body["base_currency"],
                        benchmark=body["benchmark"],
                        is_benchmark=bool(spec.get("is_benchmark")),
                        lot_method=body["lot_method"],
                        current_version_no=1,
                    )
                ).inserted_primary_key[0]
            )
        except IntegrityError as exc:
            raise PortfolioError(f"A portfolio named “{name}” already exists") from exc
        _insert_version(conn, pid, 1, body, spec.get("note"), draft=False)
    return pid


def _insert_version(
    conn: Any, pid: int, number: int, body: dict[str, Any], note: Any, draft: bool
) -> None:
    vid = conn.execute(
        insert(portfolio_versions).values(
            portfolio_id=pid,
            version_no=number,
            note=note,
            rebalance=body["rebalance"],
            cost_bps=body["cost_bps"],
            initial_value=body["initial_value"],
            is_draft=draft,
            start_date=date.fromisoformat(body["start_date"]) if body["start_date"] else None,
            definition_hash=definition_hash(body),
        )
    ).inserted_primary_key[0]
    if body["holdings"]:
        conn.execute(
            insert(portfolio_targets), [{"version_id": vid, **h} for h in body["holdings"]]
        )


def save_version(engine: Engine, pid: int, spec: dict[str, Any], draft: bool = False) -> int:
    """New immutable version; becomes current unless it is a what-if draft (P-014)."""
    current = load_portfolio(engine, pid)
    merged = {
        k: current[k]
        for k in (
            "mode",
            "base_currency",
            "benchmark",
            "lot_method",
            "holdings",
            "rebalance",
            "cost_bps",
            "start_date",
            "initial_value",
        )
    } | spec
    _validate(merged)
    body = _version_body(merged)
    with engine.begin() as conn:
        number = (
            int(
                conn.execute(
                    select(func.max(portfolio_versions.c.version_no)).where(
                        portfolio_versions.c.portfolio_id == pid
                    )
                ).scalar_one()
            )
            + 1
        )
        _insert_version(conn, pid, number, body, spec.get("note"), draft)
        if not draft:
            values: dict[str, Any] = {
                "current_version_no": number,
                "mode": body["mode"],
                "base_currency": body["base_currency"],
                "benchmark": body["benchmark"],
                "lot_method": body["lot_method"],
                "updated_at": func.now(),
            }
            if spec.get("name"):
                values["name"] = str(spec["name"]).strip()
            if "is_benchmark" in spec:
                values["is_benchmark"] = bool(spec["is_benchmark"])
            conn.execute(update(portfolios).where(portfolios.c.id == pid).values(**values))
    return number


def list_portfolios(engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(portfolios)
                .where(portfolios.c.archived_at.is_(None))
                .order_by(portfolios.c.name)
            )
            .mappings()
            .all()
        )
        counts = dict(
            conn.execute(
                select(transactions.c.portfolio_id, func.count())
                .where(transactions.c.voided_at.is_(None))
                .group_by(transactions.c.portfolio_id)
            ).all()
        )
    out = []
    for r in rows:
        d = load_portfolio(engine, int(r["id"]))
        out.append(
            {
                "id": r["id"],
                "name": r["name"],
                "mode": r["mode"],
                "base_currency": r["base_currency"],
                "benchmark": r["benchmark"],
                "is_benchmark": r["is_benchmark"],
                "owner_id": r["owner_id"],
                "shared": r["shared"],
                "version": r["current_version_no"],
                "holdings": len(d["holdings"]) or len(d["symbols"]),
                "transactions": int(counts.get(r["id"], 0)),
                "updated_at": r["updated_at"],
            }
        )
    return out


def load_portfolio(engine: Engine, pid: int, version_no: int | None = None) -> dict[str, Any]:
    """Definition of a version (default: current) + ledger metadata; transactions as a frame."""
    with engine.connect() as conn:
        p = conn.execute(select(portfolios).where(portfolios.c.id == pid)).mappings().one_or_none()
        if p is None or p["archived_at"] is not None:
            raise PortfolioError(f"Portfolio {pid} not found")
        number = version_no or int(p["current_version_no"])
        v = (
            conn.execute(
                select(portfolio_versions).where(
                    (portfolio_versions.c.portfolio_id == pid)
                    & (portfolio_versions.c.version_no == number)
                )
            )
            .mappings()
            .one_or_none()
        )
        if v is None:
            raise PortfolioError(f"Portfolio {pid} has no version {number}")
        holdings = [
            {
                "symbol": h["symbol"],
                "weight": h["weight"],
                "is_cash": h["is_cash"],
                "locked": h["locked"],
                "note": h["note"],
            }
            for h in conn.execute(
                select(portfolio_targets)
                .where(portfolio_targets.c.version_id == v["id"])
                .order_by(portfolio_targets.c.id)
            ).mappings()
        ]
        tx = pd.DataFrame(
            conn.execute(
                select(transactions)
                .where(transactions.c.portfolio_id == pid)
                .order_by(transactions.c.id)
            )
            .mappings()
            .all()
        )
    live = tx.loc[tx["voided_at"].isna()] if not tx.empty else tx
    last_id = int(tx["id"].max()) if not tx.empty else 0
    voided = int(tx["voided_at"].notna().sum()) if not tx.empty else 0
    ledger_tag = f"{len(live)}:{last_id}:{voided}"  # part of the analyses' data_version
    symbols = sorted({str(s) for s in live["symbol"].dropna()}) if not live.empty else []
    return {
        "id": pid,
        "name": p["name"],
        "mode": p["mode"],
        "base_currency": p["base_currency"],
        "benchmark": p["benchmark"],
        "is_benchmark": p["is_benchmark"],
        "lot_method": p["lot_method"],
        "version": number,
        "current_version": p["current_version_no"],
        "is_draft": v["is_draft"],
        "hash": v["definition_hash"],
        "note": v["note"],
        "holdings": holdings,
        "rebalance": v["rebalance"],
        "cost_bps": v["cost_bps"],
        "start_date": v["start_date"].isoformat() if v["start_date"] else None,
        "initial_value": v["initial_value"],
        "transactions": live.drop(columns=["voided_at"], errors="ignore")
        if p["mode"] == "transactions"
        else None,
        "ledger_tag": ledger_tag,
        "symbols": symbols,
    }


def list_versions(engine: Engine, pid: int) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(
                    portfolio_versions.c.version_no,
                    portfolio_versions.c.created_at,
                    portfolio_versions.c.note,
                    portfolio_versions.c.is_draft,
                    portfolio_versions.c.definition_hash,
                )
                .where(portfolio_versions.c.portfolio_id == pid)
                .order_by(portfolio_versions.c.version_no.desc())
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def archive_portfolio(engine: Engine, pid: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(portfolios).where(portfolios.c.id == pid).values(archived_at=datetime.now(UTC))
        )


# ── Ledger ───────────────────────────────────────────────────────────────────


def _clean_row(raw: dict[str, Any], line: int) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    def num(key: str, default: float | None = None) -> float | None:
        value = raw.get(key)
        if value is None or str(value).strip() == "":
            return default
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            errors.append(f"line {line}: {key} “{value}” is not a number")
            return default

    kind = str(raw.get("type") or "").strip().upper()
    if kind not in TX_TYPES:
        errors.append(f"line {line}: type “{raw.get('type')}” is not one of {sorted(TX_TYPES)}")
    try:
        trade_date = date.fromisoformat(
            str(raw.get("date") or raw.get("trade_date") or "").strip()[:10]
        )
    except ValueError:
        errors.append(f"line {line}: date “{raw.get('date')}” is not YYYY-MM-DD")
        trade_date = None
    symbol = str(raw.get("symbol") or "").strip().upper() or None
    if kind in TRADE_TYPES and not symbol:
        errors.append(f"line {line}: {kind} needs a symbol")
    currency = str(raw.get("currency") or "").strip().upper()
    if len(currency) != 3 or not currency.isalpha():
        errors.append(f"line {line}: currency “{raw.get('currency')}” is not a 3-letter code")
    quantity = num("quantity")
    if quantity is None or quantity <= 0:
        errors.append(f"line {line}: quantity must be > 0 (cash types: the amount)")
    price = num("price", 1.0)
    if kind in {"BUY", "SELL", "TRANSFER_IN", "TRANSFER_OUT"} and not price:
        errors.append(f"line {line}: {kind} needs a price")
    fees = num("fees", 0.0)
    fx = num("fx_rate") if raw.get("fx_rate") is not None else num("fx_rate_to_base")
    if errors:
        return None, errors
    return {
        "trade_date": trade_date,
        "type": kind,
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "currency": currency,
        "fees": fees,
        "fx_rate_to_base": fx,
        "account": (str(raw.get("account")).strip() or None) if raw.get("account") else None,
        "external_id": (str(raw.get("external_id")).strip() or None)
        if raw.get("external_id")
        else None,
    }, []


def parse_csv(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows of the import template (`CSV_COLUMNS`, header required) and per-line errors."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = {h.strip().lower() for h in reader.fieldnames or []}
    missing = {"date", "type", "quantity", "currency"} - header
    if missing:
        return [], [f"Missing columns: {sorted(missing)} (template: {', '.join(CSV_COLUMNS)})"]
    rows, errors = [], []
    for i, raw in enumerate(reader, start=2):
        row, errs = _clean_row({k.strip().lower(): v for k, v in raw.items() if k}, i)
        errors += errs
        if row is not None:
            rows.append(row)
    return rows, errors


def add_transactions(
    engine: Engine, pid: int, rows: list[dict[str, Any]], batch_id: int | None = None
) -> tuple[int, int]:
    """Append rows (validated with `_clean_row`); rows whose external id already exists are
    skipped (broker re-exports). Returns (inserted, duplicates)."""
    cleaned, errors = [], []
    for i, raw in enumerate(rows, start=1):
        row, errs = _clean_row(
            raw | ({"date": raw["trade_date"]} if "trade_date" in raw else {}), i
        )
        errors += errs
        if row is not None:
            cleaned.append(row)
    if errors:
        raise PortfolioError("; ".join(errors[:10]))
    inserted = duplicates = 0
    with engine.begin() as conn:
        known = {
            r[0]
            for r in conn.execute(
                select(transactions.c.external_id).where(
                    (transactions.c.portfolio_id == pid) & transactions.c.external_id.is_not(None)
                )
            )
        }
        for row in cleaned:
            if row["external_id"] and row["external_id"] in known:
                duplicates += 1
                continue
            conn.execute(
                insert(transactions).values(portfolio_id=pid, import_batch_id=batch_id, **row)
            )
            if row["external_id"]:
                known.add(row["external_id"])
            inserted += 1
        conn.execute(update(portfolios).where(portfolios.c.id == pid).values(updated_at=func.now()))
    return inserted, duplicates


def import_csv(
    engine: Engine, pid: int, text: str, file_name: str | None, dry_run: bool
) -> dict[str, Any]:
    rows, errors = parse_csv(text)
    key = hashlib.sha256(f"{pid}:{text}".encode()).hexdigest()
    result: dict[str, Any] = {
        "rows": len(rows),
        "errors": errors,
        "dry_run": dry_run,
        "inserted": 0,
        "duplicates": 0,
        "preview": [_jsonable(r) for r in rows[:20]],
    }
    if dry_run or errors:
        return result
    with engine.begin() as conn:
        seen = conn.execute(
            select(import_batches.c.id).where(import_batches.c.idempotency_key == key)
        ).scalar_one_or_none()
        if seen is not None:
            return result | {"already_imported": True, "batch_id": seen}
        batch_id = int(
            conn.execute(
                insert(import_batches).values(
                    portfolio_id=pid,
                    source="csv",
                    file_name=file_name,
                    idempotency_key=key,
                    row_count=len(rows),
                    status="imported",
                )
            ).inserted_primary_key[0]
        )
    inserted, duplicates = add_transactions(engine, pid, rows, batch_id)
    return result | {"inserted": inserted, "duplicates": duplicates, "batch_id": batch_id}


def void_transaction(engine: Engine, pid: int, tx_id: int) -> bool:
    with engine.begin() as conn:
        done = conn.execute(
            update(transactions)
            .where(
                (transactions.c.id == tx_id)
                & (transactions.c.portfolio_id == pid)
                & transactions.c.voided_at.is_(None)
            )
            .values(voided_at=datetime.now(UTC))
        ).rowcount
        if done:
            conn.execute(
                update(portfolios).where(portfolios.c.id == pid).values(updated_at=func.now())
            )
    return bool(done)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v.isoformat() if isinstance(v, date) else v for k, v in row.items()}
