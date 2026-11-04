"""agent runs (provider, model, tool path, answer, citation check, token usage)

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-24 17:00:00

Owned per user like the other user tables (0015): row-level security policies on Postgres.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
UID = "nullif(current_setting('app.user_id', true), '')::bigint"
MINE = f"(owner_id = {UID} OR (owner_id IS NULL AND current_setting('app.role', true) = 'admin'))"
READABLE = (
    f"(owner_id = {UID} OR owner_id IS NULL "
    "OR (shared AND current_setting('app.access', true) = 'read'))"
)


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", ID, autoincrement=True, nullable=False),
        sa.Column("owner_id", ID, nullable=True),
        sa.Column("shared", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("steps", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column("verification", sa.JSON(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_agent_runs_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
    )
    op.create_index(op.f("ix_agent_runs_owner_id"), "agent_runs", ["owner_id"], unique=False)
    if op.get_bind().dialect.name == "postgresql":
        for statement in (
            "ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY",
            f"CREATE POLICY agent_runs_read ON agent_runs FOR SELECT TO fda_app USING {READABLE}",
            f"CREATE POLICY agent_runs_insert ON agent_runs FOR INSERT TO fda_app "
            f"WITH CHECK {MINE}",
            f"CREATE POLICY agent_runs_update ON agent_runs FOR UPDATE TO fda_app "
            f"USING {MINE} WITH CHECK {MINE}",
            f"CREATE POLICY agent_runs_delete ON agent_runs FOR DELETE TO fda_app USING {MINE}",
        ):
            op.execute(statement)


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_runs_owner_id"), table_name="agent_runs")
    op.drop_table("agent_runs")
