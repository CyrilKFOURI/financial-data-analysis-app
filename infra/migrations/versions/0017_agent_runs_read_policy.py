"""agent runs: no system-row read access (every run has an owner)

Revision ID: 0017
Revises: 0016
Create Date: 2026-10-25 12:00:00

Agent runs always belong to the user who asked (the API refuses runs without one), so the
`owner_id IS NULL` read clause shared with reference tables is dropped for this table.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UID = "nullif(current_setting('app.user_id', true), '')::bigint"


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP POLICY IF EXISTS agent_runs_read ON agent_runs")
    op.execute(
        "CREATE POLICY agent_runs_read ON agent_runs FOR SELECT TO fda_app USING "
        f"(owner_id = {UID} OR (shared AND current_setting('app.access', true) = 'read'))"
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP POLICY IF EXISTS agent_runs_read ON agent_runs")
    op.execute(
        "CREATE POLICY agent_runs_read ON agent_runs FOR SELECT TO fda_app USING "
        f"(owner_id = {UID} OR owner_id IS NULL "
        "OR (shared AND current_setting('app.access', true) = 'read'))"
    )
