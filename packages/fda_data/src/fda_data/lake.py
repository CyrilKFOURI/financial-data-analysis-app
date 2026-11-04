"""Parquet lake (BACKEND_SCHEMA.md §5), read with DuckDB.

Layout:
  <root>/prices_daily/asset_class=<c>/instrument_id=<id>/year=<YYYY>/part.parquet
  <root>/actions/instrument_id=<id>/part.parquet
  <root>/series/code=<CODE>/part.parquet
Single writer (the ingestion worker); writes are atomic (temp file + rename).
"""

import os
import uuid
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PRICE_COLUMNS = ["session_date", "open", "high", "low", "close", "adj_close", "volume", "source"]

PRICE_SCHEMA = pa.schema(
    [
        ("session_date", pa.date32()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("adj_close", pa.float64()),
        ("volume", pa.float64()),
        ("source", pa.string()),
    ]
)


ACTION_SCHEMA = pa.schema(
    [
        ("ex_date", pa.date32()),
        ("action", pa.string()),
        ("ratio", pa.float64()),
        ("amount", pa.float64()),
        ("currency", pa.string()),
        ("source", pa.string()),
    ]
)

SERIES_SCHEMA = pa.schema([("date", pa.date32()), ("value", pa.float64()), ("source", pa.string())])


def _atomic_write(frame: pd.DataFrame, schema: pa.Schema, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".part-{uuid.uuid4().hex}.tmp"
    pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), tmp)
    os.replace(tmp, target)


def _merge_single_file(
    target: Path, schema: pa.Schema, new: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    """Union with the stored file; rows of `new` win on the same `keys`."""
    frame = new[schema.names].copy()
    if target.exists():
        existing = pq.read_table(target, schema=schema).to_pandas()
        frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=keys, keep="last")
    return frame.sort_values(keys, ignore_index=True)


class PriceLake:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _instrument_dir(self, asset_class: str, instrument_id: int) -> Path:
        return (
            self.root
            / "prices_daily"
            / f"asset_class={asset_class}"
            / f"instrument_id={instrument_id}"
        )

    def write_prices(self, asset_class: str, instrument_id: int, bars: pd.DataFrame) -> int:
        """Merge `bars` into the yearly partitions (new rows win on the same date)."""
        if bars.empty:
            return 0
        frame = bars[PRICE_COLUMNS].copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        written = 0
        for year, chunk in frame.groupby(pd.to_datetime(frame["session_date"]).dt.year):
            target_dir = self._instrument_dir(asset_class, instrument_id) / f"year={year}"
            target = target_dir / "part.parquet"
            if target.exists():
                existing = pq.read_table(target, schema=PRICE_SCHEMA).to_pandas()
                existing = existing[~existing["session_date"].isin(chunk["session_date"])]
                chunk = pd.concat([existing, chunk], ignore_index=True)
            chunk = chunk.sort_values("session_date").reset_index(drop=True)
            _atomic_write(chunk, PRICE_SCHEMA, target)
            written += len(chunk)
        return written

    def read_prices(
        self,
        asset_class: str,
        instrument_id: int,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        directory = self._instrument_dir(asset_class, instrument_id)
        if not any(directory.glob("year=*/part.parquet")):
            return pd.DataFrame(columns=PRICE_COLUMNS)
        pattern = (directory / "year=*" / "part.parquet").as_posix()
        query = (
            "SELECT session_date, open, high, low, close, adj_close, volume, source "
            "FROM read_parquet(?, hive_partitioning = false) "
            "WHERE (? IS NULL OR session_date >= ?) AND (? IS NULL OR session_date <= ?) "
            "ORDER BY session_date"
        )
        with duckdb.connect() as connection:
            return connection.execute(query, [pattern, start, start, end, end]).df()

    def price_coverage(self, asset_class: str, instrument_id: int) -> tuple[date, date, int] | None:
        """(first session, last session, rows) measured on the stored partitions."""
        directory = self._instrument_dir(asset_class, instrument_id)
        if not any(directory.glob("year=*/part.parquet")):
            return None
        pattern = (directory / "year=*" / "part.parquet").as_posix()
        query = (
            "SELECT min(session_date), max(session_date), count(*) "
            "FROM read_parquet(?, hive_partitioning = false)"
        )
        with duckdb.connect() as connection:
            first, last, rows = connection.execute(query, [pattern]).fetchone() or (None, None, 0)
        return (first, last, int(rows)) if rows else None

    def actions_coverage(self, instrument_id: int) -> tuple[date, date, int] | None:
        actions = self.read_actions(instrument_id)
        if actions.empty:
            return None
        return actions["ex_date"].min(), actions["ex_date"].max(), len(actions)

    def series_coverage(self, code: str) -> tuple[date, date, int] | None:
        series = self.read_series(code)
        if series.empty:
            return None
        first, last = pd.Timestamp(series["date"].min()), pd.Timestamp(series["date"].max())
        return first.date(), last.date(), len(series)

    def _actions_file(self, instrument_id: int) -> Path:
        return self.root / "actions" / f"instrument_id={instrument_id}" / "part.parquet"

    def write_actions(self, instrument_id: int, actions: pd.DataFrame) -> int:
        if actions.empty:
            return 0
        frame = actions.copy()
        frame["ex_date"] = pd.to_datetime(frame["ex_date"]).dt.date
        target = self._actions_file(instrument_id)
        merged = _merge_single_file(target, ACTION_SCHEMA, frame, ["ex_date", "action"])
        _atomic_write(merged, ACTION_SCHEMA, target)
        return len(actions)

    def read_actions(self, instrument_id: int) -> pd.DataFrame:
        target = self._actions_file(instrument_id)
        if not target.exists():
            return pd.DataFrame(columns=ACTION_SCHEMA.names)
        return pq.read_table(target, schema=ACTION_SCHEMA).to_pandas()

    def _series_file(self, code: str) -> Path:
        return self.root / "series" / f"code={code}" / "part.parquet"

    def write_series(self, code: str, observations: pd.DataFrame) -> int:
        if observations.empty:
            return 0
        frame = observations.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        target = self._series_file(code)
        merged = _merge_single_file(target, SERIES_SCHEMA, frame, ["date"])
        _atomic_write(merged, SERIES_SCHEMA, target)
        return len(observations)

    def read_series(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        target = self._series_file(code)
        if not target.exists():
            return pd.DataFrame(columns=SERIES_SCHEMA.names)
        query = (
            "SELECT date, value, source FROM read_parquet(?) "
            "WHERE (? IS NULL OR date >= ?) AND (? IS NULL OR date <= ?) ORDER BY date"
        )
        with duckdb.connect() as connection:
            return connection.execute(query, [target.as_posix(), start, start, end, end]).df()
