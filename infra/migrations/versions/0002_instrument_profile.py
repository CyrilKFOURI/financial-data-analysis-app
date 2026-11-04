"""instrument provider profile extras

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-23 11:40:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("instruments") as batch:
        batch.add_column(sa.Column("profile", sa.JSON(), server_default="{}", nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("instruments") as batch:
        batch.drop_column("profile")
