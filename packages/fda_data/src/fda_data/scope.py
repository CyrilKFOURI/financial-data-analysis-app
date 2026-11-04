"""Row ownership for multi-user (M31 §2): every statement on a user-owned table is scoped.

Two layers enforce the same rule:
1. **Statement rewriting (every backend).** A `before_execute` hook adds owner criteria to
   every SELECT, UPDATE and DELETE that touches a user table, including subqueries and child
   tables, and refuses child inserts under a parent the actor cannot write. No query can forget
   its filter, and the rule also holds on SQLite (tests, e2e).
2. **Postgres row-level security (defence in depth).** Migration 0015 adds policies and a
   non-owner role `fda_app`. Each transaction of an actor runs `SET LOCAL ROLE fda_app` and
   sets `app.user_id`, `app.role` and `app.access`, so a bug in layer 1 still cannot cross users.

The rule:
- *Read*: own rows, system rows (`owner_id` NULL), and other users' rows marked `shared`, but
  only in read requests.
- *Write* (UPDATE, DELETE, child INSERT): own rows only. System rows are written by admins, and
  by analysts for reference tables such as universes.
- *System work* (no actor: scheduler tick, worker, migrations) is never restricted.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import (
    Alias,
    ColumnElement,
    Delete,
    Engine,
    FromClause,
    Insert,
    Join,
    Select,
    Table,
    Update,
    and_,
    event,
    or_,
    select,
)
from sqlalchemy.engine import Connection
from sqlalchemy.sql import visitors

from fda_data.actor import Actor, current_actor
from fda_data.schema import (
    agent_actions,
    agent_runs,
    alert_rules,
    comparisons,
    import_batches,
    investor_profiles,
    notifications,
    paper_accounts,
    paper_orders,
    portfolio_targets,
    portfolio_versions,
    portfolios,
    saved_screens,
    schedule_runs,
    schedules,
    snapshots,
    strategies,
    strategy_trials,
    transactions,
    universe_members,
    universes,
)

OWNED: frozenset[Table] = frozenset(
    {
        portfolios,
        comparisons,
        alert_rules,
        snapshots,
        universes,
        saved_screens,
        investor_profiles,
        strategies,
        paper_accounts,
        agent_actions,
        schedules,
        agent_runs,
    }
)
# child table → (parent table, foreign key column); access follows the parent row
CHILDREN: dict[Table, tuple[Table, str]] = {
    portfolio_versions: (portfolios, "portfolio_id"),
    portfolio_targets: (portfolio_versions, "version_id"),
    import_batches: (portfolios, "portfolio_id"),
    transactions: (portfolios, "portfolio_id"),
    strategy_trials: (strategies, "strategy_id"),
    paper_orders: (paper_accounts, "account_id"),
    schedule_runs: (schedules, "schedule_id"),
    universe_members: (universes, "universe_id"),
    notifications: (alert_rules, "rule_id"),
}
# system rows (owner NULL) that analysts may also maintain: shared reference lists
ANALYST_SYSTEM_WRITES: frozenset[Table] = frozenset({universes})
SCOPED = OWNED | frozenset(CHILDREN)


class ScopeError(PermissionError):
    """A write under a parent row the actor does not own (reported as 404, never 403)."""


def _base(source: FromClause) -> Table | None:
    if isinstance(source, Table):
        return source
    if isinstance(source, Alias) and isinstance(source.element, Table):
        return source.element
    return None


def _owned_criteria(source: FromClause, table: Table, actor: Actor, write: bool) -> Any:
    c = source.c
    mine = c.owner_id == actor.user_id
    if write:
        system_writer = actor.role == "admin" or (
            actor.role == "analyst" and table in ANALYST_SYSTEM_WRITES
        )
        return or_(mine, and_(c.owner_id.is_(None), system_writer)) if system_writer else mine
    readable = [mine, c.owner_id.is_(None)]
    if actor.access == "read":
        readable.append(c.shared.is_(True))
    return or_(*readable)


def _allowed_ids(table: Table, actor: Actor, write: bool) -> Select[Any]:
    """Ids of `table` rows the actor may read (or write); recursive for grandchildren."""
    return select(table.c.id).where(_criteria(table, table, actor, write))


def _criteria(source: FromClause, table: Table, actor: Actor, write: bool) -> Any:
    if table in OWNED:
        return _owned_criteria(source, table, actor, write)
    parent, key = CHILDREN[table]
    return source.c[key].in_(_allowed_ids(parent, actor, write))


def _sources(froms: Sequence[FromClause]) -> list[FromClause]:
    out: list[FromClause] = []
    for item in froms:
        if isinstance(item, Join):
            out.extend(_sources([item.left, item.right]))
        else:
            out.append(item)
    return out


def scope_statement(statement: Any, actor: Actor) -> Any:
    """Copy of `statement` with owner criteria on every scoped table it reads or writes."""

    def visit_select(element: Select[Any]) -> None:
        extra: list[ColumnElement[bool]] = []
        for source in _sources(element.get_final_froms()):
            table = _base(source)
            if table is not None and table in SCOPED:
                extra.append(_criteria(source, table, actor, write=False))
        if extra:
            element._where_criteria = (*element._where_criteria, *extra)

    def visit_dml(element: Update | Delete) -> None:
        table = _base(element.table)
        if table is not None and table in SCOPED:
            crit = _criteria(element.table, table, actor, write=True)
            element._where_criteria = (*element._where_criteria, crit)

    return visitors.cloned_traverse(
        statement, {}, {"select": visit_select, "update": visit_dml, "delete": visit_dml}
    )


def _insert_parent_ids(statement: Insert, key: str, params: Any) -> set[Any]:
    values: set[Any] = set()
    for column, bound in (statement._values or {}).items():
        name = getattr(column, "key", column)
        if name == key:
            values.add(getattr(bound, "value", bound))
    for row in statement._multi_values or ():
        for item in row:
            if isinstance(item, Mapping):
                values.update(v for k, v in item.items() if getattr(k, "key", k) == key)
    rows = params if isinstance(params, list | tuple) else [params]
    for row in rows:
        if isinstance(row, Mapping) and key in row:
            values.add(row[key])
    return {v for v in values if v is not None}


def _check_child_insert(conn: Connection, statement: Insert, params: Any, actor: Actor) -> None:
    table = _base(statement.table)
    if table is None or table not in CHILDREN:
        return
    parent, key = CHILDREN[table]
    wanted = _insert_parent_ids(statement, key, params)
    if not wanted:
        return
    allowed = _allowed_ids(parent, actor, write=True).where(parent.c.id.in_(wanted))
    found = {int(row[0]) for row in conn.execute(allowed)}
    if found != {int(v) for v in wanted}:
        raise ScopeError(f"Not found: {parent.name} row")


def _before_execute(
    conn: Connection,
    clauseelement: Any,
    multiparams: Any,
    params: Any,
    execution_options: Any,
) -> tuple[Any, Any, Any]:
    actor = current_actor()
    if actor is None or execution_options.get("fda_unscoped"):
        return clauseelement, multiparams, params
    if isinstance(clauseelement, Insert):
        _check_child_insert(conn, clauseelement, multiparams or params, actor)
        return clauseelement, multiparams, params
    if isinstance(clauseelement, Select | Update | Delete):
        return scope_statement(clauseelement, actor), multiparams, params
    return clauseelement, multiparams, params


def _postgres_roles(conn: Connection) -> None:
    """Layer 2: switch each actor transaction to the RLS-bound role with its settings."""
    actor = current_actor()
    if actor is None or conn.dialect.name != "postgresql" or os.getenv("FDA_RLS", "1") == "0":
        return
    cursor = conn.connection.dbapi_connection.cursor()  # type: ignore[union-attr]
    try:
        cursor.execute("SET LOCAL ROLE fda_app")
        cursor.execute(
            "SELECT set_config('app.user_id', %s, true), set_config('app.role', %s, true), "
            "set_config('app.access', %s, true)",
            (str(actor.user_id), actor.role, actor.access),
        )
    finally:
        cursor.close()


_installed = False


def install() -> None:
    """Register both layers on every engine (idempotent; called on import)."""
    global _installed
    if _installed:
        return
    event.listen(Engine, "before_execute", _before_execute, retval=True)
    event.listen(Engine, "begin", _postgres_roles)
    _installed = True
