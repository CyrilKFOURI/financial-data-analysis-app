"""M31 backups on SQLite: archive round trip, manifest checks, tamper detection, safe restores."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine, delete, insert, text

from fda_data import backup
from fda_data.schema import schedules


def _seed(engine: Engine, root: Path) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
        conn.execute(text("INSERT INTO alembic_version VALUES ('0014')"))
        for i in range(3):
            conn.execute(
                insert(schedules).values(
                    name=f"s{i}",
                    kind="alerts",
                    cron="0 7 * * *",
                    timezone="UTC",
                    spec={},
                    enabled=True,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
    prices = root / "lake" / "prices_daily" / "asset_class=equity" / "instrument_id=1" / "year=2025"
    prices.mkdir(parents=True)
    pd.DataFrame(
        {"session_date": pd.date_range("2025-01-02", periods=5), "close": range(5)}
    ).to_parquet(prices / "part.parquet")
    (root / "reports").mkdir()
    (root / "reports" / "weekly.html").write_text("<!doctype html>", "utf-8")
    (root / "backups").mkdir()  # never archived


def test_round_trip_and_drill(engine: Engine, tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(engine, root)
    url = engine.url.render_as_string(hide_password=False)
    made = backup.create(root / "backups", keep=2, url=url, root=root)
    assert made["files"] == 2
    assert made["rows"] >= 4

    target = tmp_path / "restored"
    target_url = f"sqlite:///{(target / 'db.sqlite').as_posix()}"
    backup.restore(Path(made["file"]), target_url, target)
    report = backup.verify(Path(made["file"]), target_url, target)
    assert report.ok, report.mismatches()
    assert (target / "reports" / "weekly.html").exists()
    assert not (target / "backups").exists()

    drilled = backup.drill(Path(made["file"]), url)
    assert drilled.ok, drilled.mismatches()

    # a second restore over the now non-empty target is refused
    with pytest.raises(backup.BackupError, match="not empty"):
        backup.restore(Path(made["file"]), target_url, target)


def test_mismatches_are_reported(engine: Engine, tmp_path: Path) -> None:
    root = tmp_path / "data"
    _seed(engine, root)
    made = backup.create(root / "backups", url=engine.url.render_as_string(), root=root)
    target = tmp_path / "restored"
    target_url = f"sqlite:///{(target / 'db.sqlite').as_posix()}"
    backup.restore(Path(made["file"]), target_url, target)

    (target / "reports" / "weekly.html").write_text("tampered", "utf-8")
    restored = backup.create_engine(target_url)
    with restored.begin() as conn:
        conn.execute(delete(schedules).where(schedules.c.name == "s0"))
    restored.dispose()
    report = backup.verify(Path(made["file"]), target_url, target)
    assert not report.ok
    failed = {c["check"] for c in report.mismatches()}
    assert failed == {"rows schedules", "file checksums"}


def test_retention_keeps_newest(tmp_path: Path) -> None:
    for stamp in ("20250101T000000Z", "20250102T000000Z", "20250103T000000Z"):
        (tmp_path / f"fda-backup-{stamp}.tar.gz").write_bytes(b"x")
    assert backup.prune(tmp_path, keep=2) == ["fda-backup-20250101T000000Z.tar.gz"]
    assert backup.latest(tmp_path) == tmp_path / "fda-backup-20250103T000000Z.tar.gz"


def test_backend_mismatch_and_missing_tool(
    engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    _seed(engine, root)
    made = backup.create(root / "backups", url=engine.url.render_as_string(), root=root)
    monkeypatch.setattr(backup, "_empty_database", lambda url: True)
    with pytest.raises(backup.BackupError, match="cannot restore"):
        backup.restore(Path(made["file"]), "postgresql+psycopg://u@127.0.0.1:1/x", tmp_path / "n")
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("FDA_PG_BIN", raising=False)
    with pytest.raises(backup.BackupError, match="pg_dump not found"):
        backup._pg_tool("pg_dump")
