from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine

from fda_data.lake import PriceLake
from fda_data.schema import metadata


@pytest.fixture(autouse=True)
def offline_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """`default_chain()` must never reach the network in tests (fixtures replay real data)."""
    monkeypatch.setenv("FDA_PROVIDERS", "fake")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Throw-away SQLite database with the full schema (Postgres is exercised in e2e)."""
    db = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    metadata.create_all(db)
    yield db
    db.dispose()


@pytest.fixture
def lake(tmp_path: Path) -> PriceLake:
    return PriceLake(tmp_path / "lake")
