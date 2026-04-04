"""add cppp and ocds transaction sources

Revision ID: 0002_cppp_ocds_src
Revises: 0001_initial_schema
Create Date: 2026-03-20 12:16:00
"""

from __future__ import annotations

from alembic import op


revision = "0002_cppp_ocds_src"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'transaction_source_enum' AND e.enumlabel = 'CPPP'
            ) THEN
                ALTER TYPE transaction_source_enum ADD VALUE 'CPPP';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = 'transaction_source_enum' AND e.enumlabel = 'OCDS'
            ) THEN
                ALTER TYPE transaction_source_enum ADD VALUE 'OCDS';
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    # PostgreSQL enum value removal is non-trivial and unsafe in-place.
    # Keeping downgrade as no-op to avoid destructive type rebuilds.
    pass
