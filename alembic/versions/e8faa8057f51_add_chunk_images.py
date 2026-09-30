"""add chunk images

Revision ID: e8faa8057f51
Revises: 1407c23cafbe
Create Date: 2026-09-16 09:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = 'e8faa8057f51'
down_revision: str | None = '1407c23cafbe'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('images', sa.JSON(), nullable=False, server_default='[]'))


def downgrade() -> None:
    with op.batch_alter_table('chunks', schema=None) as batch_op:
        batch_op.drop_column('images')
