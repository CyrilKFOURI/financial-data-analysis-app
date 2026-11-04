"""M31 §2 defence in depth: Postgres RLS alone isolates users (statement rewriting turned off).

Needs a Postgres 16 server: FDA_TEST_PG_URL=postgresql+psycopg://fda:…@127.0.0.1:5433/postgres.
A scratch database is migrated to head, used and dropped.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, func, insert, select, text, update
from sqlalchemy.engine import make_url

from fda_data import scope
from fda_data.actor import Actor, acting_as
from fda_data.schema import paper_accounts, paper_orders, portfolios, users

PG = os.getenv("FDA_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG, reason="set FDA_TEST_PG_URL to run the Postgres RLS tests")
SCRATCH = "fda_rls_test"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def pg(monkeypatch_module: pytest.MonkeyPatch) -> Iterator[Engine]:
    server = make_url(PG or "")
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {SCRATCH}"))
    url = server.set(database=SCRATCH).render_as_string(hide_password=False)
    monkeypatch_module.setenv("DATABASE_URL", url)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "infra" / "migrations"))
    command.upgrade(config, "head")
    engine = create_engine(url)
    scope.install()
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH} WITH (FORCE)"))
        admin.dispose()


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def test_rls_without_the_statement_layer(pg: Engine) -> None:
    with pg.begin() as conn:  # system: no actor, table owner, RLS not applied
        conn.execute(insert(users).values(id=2, email="a@x", display_name="A", role="analyst"))
        conn.execute(insert(users).values(id=3, email="b@x", display_name="B", role="analyst"))
        conn.execute(text("SELECT setval(pg_get_serial_sequence('users', 'id'), 3)"))
        for owner, name, shared in ((2, "A", False), (3, "B", False), (3, "B shared", True)):
            conn.execute(
                insert(portfolios).values(name=name, mode="weights", owner_id=owner, shared=shared)
            )
        account = conn.execute(
            insert(paper_accounts)
            .values(name="B acct", currency="USD", starting_cash=1, costs={}, owner_id=3)
            .returning(paper_accounts.c.id)
        ).scalar_one()

    raw = {"fda_unscoped": True}  # layer 1 off: only the database policies decide
    with acting_as(Actor(2, "analyst", "read")), pg.connect() as conn:
        conn = conn.execution_options(**raw)
        names = set(conn.execute(select(portfolios.c.name)).scalars())
        assert names == {"A", "B shared"}
        assert conn.execute(select(func.count()).select_from(paper_orders)).scalar() == 0
    with acting_as(Actor(2, "analyst", "write")), pg.begin() as conn:
        conn = conn.execution_options(**raw)
        assert set(conn.execute(select(portfolios.c.name)).scalars()) == {"A"}
        assert conn.execute(update(portfolios).values(lot_method="fifo")).rowcount == 1
    with (
        pytest.raises(Exception, match="row-level security"),
        acting_as(Actor(2, "analyst")),
        pg.begin() as conn,
    ):
        conn.execution_options(**raw).execute(
            insert(paper_orders).values(
                account_id=account,
                symbol="X",
                side="buy",
                quantity=1,
                order_type="market",
                tif="DAY",
                decision_date="2025-01-02",
                decision_price=1,
                source="test",
            )
        )
    with pg.connect() as conn:  # system still sees everything
        assert conn.execute(select(func.count()).select_from(portfolios)).scalar() == 3
