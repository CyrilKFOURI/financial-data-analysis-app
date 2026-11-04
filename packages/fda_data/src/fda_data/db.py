"""Database & lake locations from the environment (.env, see .env.example)."""

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine

DEFAULT_DATABASE_URL = "postgresql+psycopg://fda:fda_local_only@127.0.0.1:5433/fda"


def database_url() -> str:
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url(), pool_pre_ping=True, future=True)


def lake_root() -> Path:
    return Path(os.getenv("DATA_DIR") or "data").resolve() / "lake"
