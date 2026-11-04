from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from fda_api.main import app
from fda_api.security import CLIENT_HEADER
from fda_data.schema import metadata


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """API client on a throw-away SQLite DB and lake, in-memory cache, inline jobs."""
    database_url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    setup_engine = create_engine(database_url)
    metadata.create_all(setup_engine)
    setup_engine.dispose()
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("FDA_PROVIDERS", "fake")  # no network in tests
    for key in (
        "ANTHROPIC",
        "GEMINI",
        "OPENAI",
        "MISTRAL",
        "DEEPSEEK",
        "GROQ",
        "XAI",
        "OPENROUTER",
        "CUSTOM",
    ):
        monkeypatch.delenv(f"{key}_API_KEY", raising=False)  # tests never call a paid LLM
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    with TestClient(app, headers={CLIENT_HEADER: "test"}) as test_client:
        yield test_client
