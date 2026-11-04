"""users, sessions, row ownership and Postgres row-level security (M31 §2)

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-24 14:00:00

Existing user rows go to user 1 (the built-in local owner, adopted by the first admin in accounts
mode); reference universes (index, crypto) become system rows. On Postgres, RLS policies bind the
`fda_app` role that the API switches to for each request (`fda_data.scope`).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
OWNED = (
    "portfolios",
    "comparisons",
    "alert_rules",
    "snapshots",
    "universes",
    "saved_screens",
    "investor_profiles",
    "strategies",
    "paper_accounts",
    "agent_actions",
    "schedules",
)
UNIQUE_NAMES = (
    "portfolios",
    "comparisons",
    "universes",
    "strategies",
    "paper_accounts",
    "schedules",
)
CHILDREN = {
    "portfolio_versions": ("portfolios", "portfolio_id"),
    "portfolio_targets": ("portfolio_versions", "version_id"),
    "import_batches": ("portfolios", "portfolio_id"),
    "transactions": ("portfolios", "portfolio_id"),
    "strategy_trials": ("strategies", "strategy_id"),
    "paper_orders": ("paper_accounts", "account_id"),
    "schedule_runs": ("schedules", "schedule_id"),
    "universe_members": ("universes", "universe_id"),
    "notifications": ("alert_rules", "rule_id"),
}
UID = "nullif(current_setting('app.user_id', true), '')::bigint"
ROLE = "current_setting('app.role', true)"
ACCESS = "current_setting('app.access', true)"


def _system_writer(table: str) -> str:
    roles = "('admin', 'analyst')" if table == "universes" else "('admin')"
    return f"(owner_id IS NULL AND {ROLE} IN {roles})"


def _writable_ids(table: str) -> str:
    if table in CHILDREN:
        parent, key = CHILDREN[table]
        return f"SELECT id FROM {table} WHERE {key} IN ({_writable_ids(parent)})"
    return f"SELECT id FROM {table} WHERE owner_id = {UID} OR {_system_writer(table)}"


def _policies() -> list[str]:
    sql = [
        "DO $$ BEGIN CREATE ROLE fda_app NOLOGIN; "
        "EXCEPTION WHEN duplicate_object THEN NULL; END $$",
        "GRANT fda_app TO CURRENT_USER",
        "GRANT USAGE ON SCHEMA public TO fda_app",
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fda_app",
        "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fda_app",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fda_app",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO fda_app",
    ]
    for table in OWNED:
        mine = f"(owner_id = {UID} OR {_system_writer(table)})"
        readable = f"(owner_id = {UID} OR owner_id IS NULL OR (shared AND {ACCESS} = 'read'))"
        sql += [
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
            f"CREATE POLICY {table}_read ON {table} FOR SELECT TO fda_app USING {readable}",
            f"CREATE POLICY {table}_insert ON {table} FOR INSERT TO fda_app WITH CHECK {mine}",
            f"CREATE POLICY {table}_update ON {table} FOR UPDATE TO fda_app "
            f"USING {mine} WITH CHECK {mine}",
            f"CREATE POLICY {table}_delete ON {table} FOR DELETE TO fda_app USING {mine}",
        ]
    for table, (parent, key) in CHILDREN.items():
        readable = f"({key} IN (SELECT id FROM {parent}))"  # the parent's own policy applies
        writable = f"({key} IN ({_writable_ids(parent)}))"
        sql += [
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
            f"CREATE POLICY {table}_read ON {table} FOR SELECT TO fda_app USING {readable}",
            f"CREATE POLICY {table}_insert ON {table} FOR INSERT TO fda_app WITH CHECK {writable}",
            f"CREATE POLICY {table}_update ON {table} FOR UPDATE TO fda_app "
            f"USING {writable} WITH CHECK {writable}",
            f"CREATE POLICY {table}_delete ON {table} FOR DELETE TO fda_app USING {writable}",
        ]
    return sql


def upgrade() -> None:
    users = op.create_table(
        "users",
        sa.Column("id", ID, autoincrement=True, nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("user_id", ID, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sessions")),
    )
    op.create_index("ix_sessions_user", "sessions", ["user_id"], unique=False)
    assert users is not None
    op.bulk_insert(
        users,
        [{"id": 1, "email": "local", "display_name": "Local user", "role": "admin"}],
    )
    bind = op.get_bind()
    postgres = bind.dialect.name == "postgresql"
    if postgres:
        op.execute("SELECT setval(pg_get_serial_sequence('users', 'id'), 1)")

    for table in OWNED:
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("owner_id", ID, nullable=True))
            batch.add_column(sa.Column("shared", sa.Boolean(), server_default="0", nullable=False))
            batch.create_index(op.f(f"ix_{table}_owner_id"), ["owner_id"], unique=False)
            batch.create_foreign_key(
                op.f(f"fk_{table}_owner_id_users"),
                "users",
                ["owner_id"],
                ["id"],
                ondelete="CASCADE",
            )
            if table in UNIQUE_NAMES:
                batch.drop_constraint(f"uq_{table}_name", type_="unique")
                batch.create_unique_constraint(f"uq_{table}_owner_id", ["owner_id", "name"])
        if table == "universes":
            op.execute("UPDATE universes SET owner_id = 1 WHERE kind NOT IN ('index', 'crypto')")
        else:
            op.execute(f"UPDATE {table} SET owner_id = 1")

    if postgres:
        for statement in _policies():
            op.execute(statement)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table in (*OWNED, *CHILDREN):
            for action in ("read", "insert", "update", "delete"):
                op.execute(f"DROP POLICY IF EXISTS {table}_{action} ON {table}")
            op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    for table in OWNED:
        with op.batch_alter_table(table) as batch:
            if table in UNIQUE_NAMES:
                batch.drop_constraint(f"uq_{table}_owner_id", type_="unique")
                batch.create_unique_constraint(f"uq_{table}_name", ["name"])
            batch.drop_constraint(op.f(f"fk_{table}_owner_id_users"), type_="foreignkey")
            batch.drop_index(op.f(f"ix_{table}_owner_id"))
            batch.drop_column("shared")
            batch.drop_column("owner_id")
    op.drop_index("ix_sessions_user", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("users")
