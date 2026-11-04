"""M31 §4 backups: database dump + Parquet lake + reports in one archive, restore and drill.

An archive `fda-backup-<UTC stamp>.tar.gz` holds `db.dump` (`pg_dump -Fc`, or the SQLite file),
the `lake/` and `reports/` trees of DATA_DIR and a `manifest.json`. The manifest records the
alembic revision, the row count of every table and the SHA-256 of every file. On Postgres, the
counts and the dump read the same exported snapshot, so they agree even while the app writes.

Restores only go to an empty database and an empty data directory (never over live data);
`verify` compares a restore with its manifest and `drill` runs restore + verify in a scratch
database that it drops afterwards.

CLI: `python -m fda_data.backup create|restore|verify|drill` (see `--help`).
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Connection, create_engine, func, inspect, select, table, text
from sqlalchemy.engine import URL, make_url

from fda_data.actor import acting_as
from fda_data.db import database_url, lake_root

FORMAT = 1
PREFIX = "fda-backup-"
TREES = ("lake", "reports")
DRILL_DB = "fda_restore_drill"


class BackupError(RuntimeError):
    pass


@dataclass
class Report:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, expected: Any, actual: Any) -> None:
        passed = expected == actual
        self.checks.append({"check": name, "expected": expected, "actual": actual, "ok": passed})
        self.ok = self.ok and passed

    def mismatches(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["ok"]]


def data_dir() -> Path:
    return lake_root().parent


def _is_sqlite(url: URL) -> bool:
    return url.get_backend_name() == "sqlite"


def _sqlite_path(url: URL) -> Path:
    if not url.database or url.database == ":memory:":
        raise BackupError("In-memory SQLite databases cannot be backed up")
    return Path(url.database)


def _pg_env(url: URL) -> tuple[list[str], dict[str, str]]:
    """libpq connection flags; the password goes through the environment, never argv."""
    args = ["--host", url.host or "127.0.0.1", "--port", str(url.port or 5432)]
    if url.username:
        args += ["--username", url.username]
    env = dict(os.environ)
    if url.password:
        env["PGPASSWORD"] = str(url.password)
    return args, env


def _pg_tool(name: str) -> str:
    folder = os.getenv("FDA_PG_BIN")
    tool = str(Path(folder) / name) if folder else shutil.which(name)
    if not tool:
        raise BackupError(f"{name} not found: install postgresql-client-16 or set FDA_PG_BIN")
    return tool


def _counts(conn: Connection) -> dict[str, int]:
    names = sorted(inspect(conn).get_table_names())
    return {
        n: int(conn.execute(select(func.count()).select_from(table(n))).scalar_one()) for n in names
    }


def _revision(conn: Connection) -> str | None:
    if "alembic_version" not in inspect(conn).get_table_names():
        return None
    return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tree in TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            out[rel] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return out


# ── create ───────────────────────────────────────────────────────────────────


def _create(
    out_dir: Path, keep: int = 14, url: str | None = None, root: Path | None = None
) -> dict[str, Any]:
    """Write one archive to `out_dir` and prune it down to the newest `keep` archives."""
    db_url = make_url(url or database_url())
    root = root or data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / f"{PREFIX}{stamp}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="fda-backup-") as tmp:
        dump = Path(tmp) / "db.dump"
        counts, revision = _dump(db_url, dump)
        files = _files(root)
        manifest = {
            "format": FORMAT,
            "created_at": datetime.now(UTC).isoformat(),
            "backend": db_url.get_backend_name(),
            "alembic_revision": revision,
            "tables": counts,
            "files": files,
        }
        (Path(tmp) / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
        partial = target.with_suffix(".partial")
        with tarfile.open(partial, "w:gz") as tar:
            tar.add(Path(tmp) / "manifest.json", "manifest.json")
            tar.add(dump, "db.dump")
            for rel in files:
                tar.add(root / rel, rel)
        partial.replace(target)
    pruned = prune(out_dir, keep)
    return {
        "file": str(target),
        "bytes": target.stat().st_size,
        "tables": len(counts),
        "rows": sum(counts.values()),
        "files": len(files),
        "pruned": pruned,
    }


def _dump(url: URL, dump: Path) -> tuple[dict[str, int], str | None]:
    if _is_sqlite(url):
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                counts, revision = _counts(conn), _revision(conn)
                # SQLite online backup API: a consistent copy even with other connections open
                raw = conn.connection.dbapi_connection
                with contextlib.closing(sqlite3.connect(dump)) as copy:
                    raw.backup(copy)  # type: ignore[union-attr]
        finally:
            engine.dispose()
        return counts, revision
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="REPEATABLE READ")
            with conn.begin():
                snapshot = conn.execute(text("SELECT pg_export_snapshot()")).scalar_one()
                counts, revision = _counts(conn), _revision(conn)
                args, env = _pg_env(url)
                done = subprocess.run(
                    [
                        _pg_tool("pg_dump"),
                        *args,
                        "--format=custom",
                        f"--snapshot={snapshot}",
                        "--no-owner",
                        "--file",
                        str(dump),
                        url.database or "postgres",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if done.returncode != 0:
                    raise BackupError(f"pg_dump failed: {done.stderr.strip()[:500]}")
    finally:
        engine.dispose()
    return counts, revision


def prune(out_dir: Path, keep: int) -> list[str]:
    archives = sorted(out_dir.glob(f"{PREFIX}*.tar.gz"))
    doomed = archives[: max(0, len(archives) - max(keep, 1))]
    for path in doomed:
        path.unlink()
    return [p.name for p in doomed]


def latest(out_dir: Path) -> Path | None:
    archives = sorted(out_dir.glob(f"{PREFIX}*.tar.gz"))
    return archives[-1] if archives else None


# ── restore / verify ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def _opened(archive: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="fda-restore-") as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")  # refuses absolute paths and `..` members
        manifest = json.loads((Path(tmp) / "manifest.json").read_text("utf-8"))
        if manifest.get("format") != FORMAT:
            raise BackupError(f"Unsupported backup format {manifest.get('format')}")
        yield Path(tmp), manifest


def _empty_database(url: URL) -> bool:
    if _is_sqlite(url):
        path = _sqlite_path(url)
        return not path.exists() or path.stat().st_size == 0
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return not inspect(conn).get_table_names()
    finally:
        engine.dispose()


def _restore(archive: Path, url: str, root: Path) -> dict[str, Any]:
    """Restore into an empty database and an empty data directory; refuses anything else."""
    db_url = make_url(url)
    if not _empty_database(db_url):
        raise BackupError("Target database is not empty: restore only into a fresh database")
    if any((root / tree).exists() and any((root / tree).iterdir()) for tree in TREES):
        raise BackupError(f"Target data directory {root} already holds a lake or reports")
    with _opened(archive) as (tmp, manifest):
        if manifest["backend"] != db_url.get_backend_name():
            raise BackupError(
                f"Backup of a {manifest['backend']} database cannot restore into "
                f"{db_url.get_backend_name()}"
            )
        if _is_sqlite(db_url):
            path = _sqlite_path(db_url)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp / "db.dump", path)
        else:
            args, env = _pg_env(db_url)
            done = subprocess.run(
                [
                    _pg_tool("pg_restore"),
                    *args,
                    "--no-owner",
                    "--exit-on-error",
                    "--dbname",
                    db_url.database or "postgres",
                    str(tmp / "db.dump"),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                raise BackupError(f"pg_restore failed: {done.stderr.strip()[:500]}")
        for rel in manifest["files"]:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp / rel, dest)
    return {"tables": len(manifest["tables"]), "files": len(manifest["files"])}


def _verify(archive: Path, url: str, root: Path) -> Report:
    """Compare a restored database and data directory with the archive's manifest."""
    with _opened(archive) as (_, manifest):
        pass
    report = Report(ok=True)
    engine = create_engine(make_url(url))
    try:
        with engine.connect() as conn:
            report.add("alembic revision", manifest["alembic_revision"], _revision(conn))
            actual = _counts(conn)
    finally:
        engine.dispose()
    for name, rows in manifest["tables"].items():
        report.add(f"rows {name}", rows, actual.get(name))
    files = _files(root)
    report.add("file count", len(manifest["files"]), len(files))
    bad = sorted(rel for rel, meta in manifest["files"].items() if files.get(rel) != meta)
    report.add("file checksums", [], bad[:20])
    prices = sorted((root / "lake" / "prices_daily").rglob("part.parquet"))
    if prices:
        frame = pd.read_parquet(prices[0])
        report.add("sample price file readable", True, len(frame) > 0)
    return report


def _drill(archive: Path, url: str | None = None) -> Report:
    """Restore drill: restore + verify into a scratch database and data dir, then drop both."""
    source = make_url(url or database_url())
    with tempfile.TemporaryDirectory(prefix="fda-drill-") as tmp:
        root = Path(tmp) / "data"
        if _is_sqlite(source):
            scratch = source.set(database=str(Path(tmp) / "drill.db"))
            _restore(archive, scratch.render_as_string(hide_password=False), root)
            return _verify(archive, scratch.render_as_string(hide_password=False), root)
        scratch = source.set(database=DRILL_DB)
        admin = create_engine(source, isolation_level="AUTOCOMMIT")
        try:
            with admin.connect() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS {DRILL_DB} WITH (FORCE)"))
                conn.execute(text(f"CREATE DATABASE {DRILL_DB}"))
            try:
                rendered = scratch.render_as_string(hide_password=False)
                _restore(archive, rendered, root)
                return _verify(archive, rendered, root)
            finally:
                with admin.connect() as conn:
                    conn.execute(text(f"DROP DATABASE IF EXISTS {DRILL_DB} WITH (FORCE)"))
        finally:
            admin.dispose()


def create(
    out_dir: Path, keep: int = 14, url: str | None = None, root: Path | None = None
) -> dict[str, Any]:
    """Write one archive to `out_dir` and prune it down to the newest `keep` archives."""
    with acting_as(None):  # a backup covers every user's rows, whoever triggers it
        return _create(out_dir, keep, url, root)


def restore(archive: Path, url: str, root: Path) -> dict[str, Any]:
    """Restore into an empty database and an empty data directory; refuses anything else."""
    with acting_as(None):
        return _restore(archive, url, root)


def verify(archive: Path, url: str, root: Path) -> Report:
    """Compare a restored database and data directory with the archive's manifest."""
    with acting_as(None):
        return _verify(archive, url, root)


def drill(archive: Path, url: str | None = None) -> Report:
    """Restore drill: restore + verify into a scratch database and data dir, then drop both."""
    with acting_as(None):
        return _drill(archive, url)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fda_data.backup", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create", help="back up DATABASE_URL and DATA_DIR")
    c.add_argument("--out", type=Path, default=None, help="default: DATA_DIR/backups")
    c.add_argument("--keep", type=int, default=14)
    for name in ("restore", "verify"):
        p = sub.add_parser(name, help=f"{name} an archive (target: --database-url, --data-dir)")
        p.add_argument("archive", type=Path)
        p.add_argument("--database-url", required=True)
        p.add_argument("--data-dir", type=Path, required=True)
    d = sub.add_parser("drill", help="restore + verify into a scratch database, then drop it")
    d.add_argument("archive", type=Path, nargs="?", help="default: newest in DATA_DIR/backups")
    args = parser.parse_args(argv)

    if args.command == "create":
        print(json.dumps(create(args.out or data_dir() / "backups", args.keep), indent=1))
        return 0
    if args.command == "restore":
        print(json.dumps(restore(args.archive, args.database_url, args.data_dir), indent=1))
        return 0
    if args.command == "verify":
        report = verify(args.archive, args.database_url, args.data_dir)
    else:
        archive = args.archive or latest(data_dir() / "backups")
        if archive is None:
            print("No backup archive found", file=sys.stderr)
            return 2
        report = drill(archive)
    print(
        json.dumps(
            {"ok": report.ok, "mismatches": report.mismatches(), "checks": len(report.checks)},
            indent=1,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
