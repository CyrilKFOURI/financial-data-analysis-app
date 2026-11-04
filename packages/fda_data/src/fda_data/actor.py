"""Who the current database work is for (M31 §2).

The API sets the actor per request (`acting_as`); background work (scheduler tick, worker,
migrations, CLI) runs with no actor, i.e. as the system. `fda_data.scope` reads the actor to
restrict user-owned rows, and new user-owned rows take `owner_id` from it (column default).
"""

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

Role = Literal["viewer", "analyst", "admin"]
Access = Literal["read", "write"]
LOCAL_USER_ID = 1  # the built-in single-user owner (migration 0015)


@dataclass(frozen=True, slots=True)
class Actor:
    user_id: int
    role: Role
    access: Access = "write"


_actor: ContextVar[Actor | None] = ContextVar("fda_actor", default=None)


def current_actor() -> Actor | None:
    return _actor.get()


def current_owner() -> int | None:
    """Column default for `owner_id`: the acting user, or None for system rows."""
    actor = _actor.get()
    return actor.user_id if actor else None


@contextlib.contextmanager
def acting_as(actor: Actor | None) -> Iterator[None]:
    token = _actor.set(actor)
    try:
        yield
    finally:
        _actor.reset(token)
