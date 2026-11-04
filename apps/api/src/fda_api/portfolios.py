"""Portfolio endpoints (M15 builder): definitions, versions (what-if drafts), ledger, CSV import
and weights ⇄ transactions conversion. Analyses run through /v1/analyses:run with a
`{"type": "portfolio", "id": "<id>[@<version>]"}` subject."""

from datetime import date
from typing import Any, Literal

import anyio
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from fda_api.resolve import _portfolio_inputs
from fda_core.portfolio import CASH, fx_rate
from fda_core.registry import InsufficientDataError
from fda_data.portfolios import (
    CSV_COLUMNS,
    PortfolioError,
    add_transactions,
    archive_portfolio,
    create_portfolio,
    import_csv,
    list_portfolios,
    list_versions,
    load_portfolio,
    save_version,
    void_transaction,
)

router = APIRouter(prefix="/v1", tags=["portfolios"])


class HoldingIn(BaseModel):
    symbol: str = Field(max_length=32)
    weight: float = Field(ge=-1, le=1)
    is_cash: bool = False
    locked: bool = False
    note: str | None = Field(None, max_length=200)


class Rebalance(BaseModel):
    type: Literal["calendar", "threshold", "none"] = "calendar"
    freq: Literal["M", "Q", "A"] = "M"
    band: float = Field(0.05, gt=0, le=0.5)


class PortfolioSpec(BaseModel):
    name: str | None = Field(None, max_length=80)
    mode: Literal["weights", "transactions"] | None = None
    base_currency: str | None = Field(None, pattern=r"^[A-Za-z]{3}$")
    benchmark: str | None = Field(None, max_length=40, pattern=r"^[\w.^=:-]+$")
    is_benchmark: bool | None = None
    lot_method: Literal["fifo", "lifo", "average"] | None = None
    holdings: list[HoldingIn] | None = Field(None, max_length=200)
    rebalance: Rebalance | None = None
    cost_bps: float | None = Field(None, ge=0, le=500)
    start_date: date | None = None
    initial_value: float | None = Field(None, gt=0)
    note: str | None = Field(None, max_length=500)


class VersionRequest(PortfolioSpec):
    draft: bool = False


class TransactionIn(BaseModel):
    date: date
    type: str = Field(max_length=16)
    symbol: str | None = Field(None, max_length=32)
    quantity: float
    price: float | None = None
    currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    fees: float | None = None
    fx_rate: float | None = None
    account: str | None = Field(None, max_length=64)
    external_id: str | None = Field(None, max_length=64)


class ImportRequest(BaseModel):
    csv: str = Field(max_length=5_000_000)
    file_name: str | None = Field(None, max_length=200)
    dry_run: bool = True


class ConvertRequest(BaseModel):
    to: Literal["transactions", "weights"]


def _spec(body: PortfolioSpec) -> dict[str, Any]:
    return body.model_dump(exclude_none=True, exclude={"draft"})


def _public(d: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in d.items() if k != "transactions"}
    tx = d.get("transactions")
    out["transaction_count"] = 0 if tx is None else len(tx)
    return out


def _fail(exc: Exception) -> HTTPException:
    status = 404 if "not found" in str(exc) else 422
    return HTTPException(status_code=status, detail=str(exc))


@router.get("/portfolios")
async def get_portfolios(request: Request) -> list[dict[str, Any]]:
    return await anyio.to_thread.run_sync(list_portfolios, request.app.state.engine)


@router.post("/portfolios", status_code=201)
async def post_portfolio(body: PortfolioSpec, request: Request) -> dict[str, Any]:
    try:
        pid = await anyio.to_thread.run_sync(
            create_portfolio, request.app.state.engine, _spec(body)
        )
    except PortfolioError as exc:
        raise _fail(exc) from exc
    return {"id": pid}


@router.get("/portfolios/{pid}")
async def get_portfolio(pid: int, request: Request, version: int | None = None) -> dict[str, Any]:
    engine = request.app.state.engine
    try:
        d = await anyio.to_thread.run_sync(load_portfolio, engine, pid, version)
    except PortfolioError as exc:
        raise _fail(exc) from exc
    return _public(d) | {"versions": await anyio.to_thread.run_sync(list_versions, engine, pid)}


@router.put("/portfolios/{pid}")
async def put_portfolio(pid: int, body: VersionRequest, request: Request) -> dict[str, Any]:
    """New immutable version (current unless `draft`: a what-if version, P-014)."""
    try:
        number = await anyio.to_thread.run_sync(
            save_version, request.app.state.engine, pid, _spec(body), body.draft
        )
    except PortfolioError as exc:
        raise _fail(exc) from exc
    return {"id": pid, "version": number, "draft": body.draft, "subject": f"{pid}@{number}"}


@router.delete("/portfolios/{pid}", status_code=204)
async def delete_portfolio(pid: int, request: Request) -> None:
    await anyio.to_thread.run_sync(archive_portfolio, request.app.state.engine, pid)


@router.get("/portfolios/{pid}/transactions")
async def get_transactions(pid: int, request: Request) -> list[dict[str, Any]]:
    try:
        d = await anyio.to_thread.run_sync(load_portfolio, request.app.state.engine, pid, None)
    except PortfolioError as exc:
        raise _fail(exc) from exc
    tx = d.get("transactions")
    if tx is None or tx.empty:
        return []
    keep = [
        "id",
        "trade_date",
        "type",
        "symbol",
        "quantity",
        "price",
        "currency",
        "fees",
        "fx_rate_to_base",
        "account",
        "external_id",
    ]
    rows = tx[keep].sort_values(["trade_date", "id"], ascending=False)
    return [
        {k: (v.isoformat() if isinstance(v, date) else None if v != v else v) for k, v in r.items()}
        for r in rows.to_dict("records")
    ]


@router.post("/portfolios/{pid}/transactions", status_code=201)
async def post_transactions(
    pid: int, rows: list[TransactionIn], request: Request
) -> dict[str, int]:
    payload = [r.model_dump() | {"date": r.date.isoformat()} for r in rows]
    try:
        inserted, duplicates = await anyio.to_thread.run_sync(
            add_transactions, request.app.state.engine, pid, payload
        )
    except PortfolioError as exc:
        raise _fail(exc) from exc
    return {"inserted": inserted, "duplicates": duplicates}


@router.post("/portfolios/{pid}/transactions/{tx_id}/void")
async def post_void(pid: int, tx_id: int, request: Request) -> dict[str, bool]:
    done = await anyio.to_thread.run_sync(void_transaction, request.app.state.engine, pid, tx_id)
    if not done:
        raise HTTPException(status_code=404, detail="Transaction not found or already voided")
    return {"voided": True}


@router.get("/portfolios-import/template")
async def import_template() -> dict[str, Any]:
    example = [
        "2024-01-02,DEPOSIT,,10000,1,USD,0,,,dep-1",
        "2024-01-02,BUY,AAPL,20,185.5,USD,1,,,t-1",
        "2024-03-15,DIVIDEND,AAPL,4.8,1,USD,0,,,d-1",
    ]
    return {
        "columns": list(CSV_COLUMNS),
        "csv": ",".join(CSV_COLUMNS) + "\n" + "\n".join(example) + "\n",
        "notes": (
            "Cash types (DEPOSIT, WITHDRAWAL, DIVIDEND, INTEREST, FEE) carry the amount "
            "in quantity."
        ),
    }


@router.post("/portfolios/{pid}/import")
async def post_import(pid: int, body: ImportRequest, request: Request) -> dict[str, Any]:
    try:
        return await anyio.to_thread.run_sync(
            import_csv, request.app.state.engine, pid, body.csv, body.file_name, body.dry_run
        )
    except PortfolioError as exc:
        raise _fail(exc) from exc


@router.post("/portfolios/{pid}/convert")
async def post_convert(pid: int, body: ConvertRequest, request: Request) -> dict[str, Any]:
    """weights → transactions: synthetic deposit + buys at the first common close;
    transactions → weights: snapshot of the current weights (new version either way)."""
    engine, lake = request.app.state.engine, request.app.state.lake

    def convert() -> dict[str, Any]:
        d = load_portfolio(engine, pid)
        if d["mode"] == body.to:
            raise PortfolioError(f"Already in {body.to} mode")
        pf = _portfolio_inputs(engine, lake, pid, None, {})
        if body.to == "weights":
            w = pf.end_weights()
            w = w[w.abs() > 1e-6]
            holdings = [
                {"symbol": str(s), "weight": float(v), "is_cash": s == CASH} for s, v in w.items()
            ]
            holdings[-1]["weight"] += 1 - sum(h["weight"] for h in holdings)
            number = save_version(
                engine,
                pid,
                {
                    "mode": "weights",
                    "holdings": holdings,
                    "start_date": pf.path.value.index[-1].date(),
                },
            )
            return {"version": number, "mode": "weights"}
        hv = pf.path.holdings_value
        if hv is None:
            raise PortfolioError("No holdings values to convert")
        day = hv.index[0]
        base = d["base_currency"]
        deposits: dict[str, float] = {}
        buys: list[dict[str, Any]] = []
        for s in hv.columns:
            value = float(hv.at[day, s])
            ccy = base if s == CASH else str((pf.meta.get(s) or {}).get("currency") or base)
            rate = float(fx_rate(pf.fx, ccy, base, pd.DatetimeIndex([day])).iloc[0])
            if value <= 0:
                continue
            deposits[ccy] = (
                deposits.get(ccy, 0.0) + value / rate
            )  # funded in the holding's currency
            if s == CASH:
                continue
            close = float(pf.prices[s]["close"].asof(day))
            buys.append(
                {
                    "date": day.date().isoformat(),
                    "type": "BUY",
                    "symbol": s,
                    "quantity": value / rate / close,
                    "price": close,
                    "currency": ccy,
                    "external_id": f"convert-{pid}-{s}",
                }
            )
        rows = [
            {
                "date": day.date().isoformat(),
                "type": "DEPOSIT",
                "symbol": None,
                "quantity": amount,
                "price": 1.0,
                "currency": ccy,
                "external_id": f"convert-{pid}-deposit-{ccy}",
            }
            for ccy, amount in deposits.items()
        ] + buys
        number = save_version(engine, pid, {"mode": "transactions"})
        add_transactions(engine, pid, rows)
        return {"version": number, "mode": "transactions", "transactions": len(rows)}

    try:
        return await anyio.to_thread.run_sync(convert)
    except (PortfolioError, InsufficientDataError) as exc:
        raise _fail(exc) from exc
