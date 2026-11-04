"""M31 row ownership: statement rewriting isolates users, sharing is read-only, system is free."""

from datetime import date

import pytest
from sqlalchemy import Engine, delete, func, insert, select, update

from fda_data import scope
from fda_data.actor import Actor, acting_as
from fda_data.schema import (
    portfolio_targets,
    portfolio_versions,
    portfolios,
    universe_members,
    universes,
    users,
)

scope.install()
ALICE = Actor(2, "analyst")
BOB = Actor(3, "analyst")


@pytest.fixture
def seeded(engine: Engine) -> Engine:
    with engine.begin() as conn:
        for uid, email in ((1, "local"), (2, "alice@x"), (3, "bob@x")):
            conn.execute(
                insert(users).values(id=uid, email=email, display_name=email, role="admin")
            )
    for actor, name in ((ALICE, "A"), (BOB, "B")):
        with acting_as(actor), engine.begin() as conn:
            pid = conn.execute(
                insert(portfolios).values(name=name, mode="weights")
            ).inserted_primary_key[0]
            vid = conn.execute(
                insert(portfolio_versions).values(
                    portfolio_id=pid, version_no=1, definition_hash="h"
                )
            ).inserted_primary_key[0]
            conn.execute(insert(portfolio_targets).values(version_id=vid, symbol="SPY", weight=1.0))
    with engine.begin() as conn:  # system row (no actor)
        conn.execute(insert(universes).values(name="S&P 500", kind="index"))
    return engine


def _names(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return set(conn.execute(select(portfolios.c.name)).scalars())


def test_owner_isolation(seeded: Engine) -> None:
    with seeded.connect() as conn:
        owners = dict(conn.execute(select(portfolios.c.name, portfolios.c.owner_id)).all())
    assert owners == {"A": 2, "B": 3}  # owner_id taken from the actor
    with acting_as(ALICE):
        assert _names(seeded) == {"A"}
        with seeded.connect() as conn:
            # children and grandchildren follow their parent, also through joins and subqueries
            assert conn.execute(select(func.count()).select_from(portfolio_targets)).scalar() == 1
            joined = select(portfolios.c.name).join_from(
                portfolio_versions, portfolios, portfolio_versions.c.portfolio_id == portfolios.c.id
            )
            assert set(conn.execute(joined).scalars()) == {"A"}
            sub = select(portfolio_versions.c.id).where(
                portfolio_versions.c.portfolio_id.in_(select(portfolios.c.id))
            )
            assert len(conn.execute(sub).all()) == 1
        with seeded.begin() as conn:
            hit = conn.execute(update(portfolios).values(name="X"))
            assert hit.rowcount == 1  # only her own row
            assert conn.execute(delete(portfolio_targets)).rowcount == 1
    assert _names(seeded) == {"X", "B"}  # system sees everything


def test_sharing_is_read_only_and_child_inserts_need_the_parent(seeded: Engine) -> None:
    with seeded.begin() as conn:
        conn.execute(update(portfolios).where(portfolios.c.name == "B").values(shared=True))
    with acting_as(Actor(2, "analyst", "read")):
        assert _names(seeded) == {"A", "B"}
    with acting_as(ALICE):  # write request: shared rows of others are invisible
        assert _names(seeded) == {"A"}
        with seeded.begin() as conn:
            assert (
                conn.execute(
                    update(portfolios).where(portfolios.c.name == "B").values(name="Z")
                ).rowcount
                == 0
            )
        bob_portfolio = 2
        with pytest.raises(scope.ScopeError), seeded.begin() as conn:
            conn.execute(
                insert(portfolio_versions).values(
                    portfolio_id=bob_portfolio, version_no=2, definition_hash="h"
                )
            )


def test_system_rows(seeded: Engine) -> None:
    with acting_as(ALICE), seeded.begin() as conn:
        uid = conn.execute(select(universes.c.id)).scalar_one()  # system rows are readable
        conn.execute(  # analysts maintain reference universes
            insert(universe_members).values(
                universe_id=uid, symbol="AAPL", valid_from=date(2025, 1, 2)
            )
        )
    with acting_as(Actor(3, "viewer")), seeded.begin() as conn:
        assert conn.execute(update(universes).values(kind="custom")).rowcount == 0
