"""SQLAlchemy ORM models for anomaly detection service."""

from __future__ import annotations

from datetime import UTC, datetime

from config import settings
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_SCHEMA = settings.db_schema


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    """Always return timezone-aware UTC datetime — never naive."""
    return datetime.now(UTC)


class AnomalySession(Base):
    """One row per anomaly detection analyze call."""

    __tablename__ = "anomaly_sessions"
    __table_args__ = {"schema": _SCHEMA}

    id: Mapped[int | None] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    source_name: Mapped[str | None] = mapped_column(String(255))
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    anomalies_found: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_rate: Mapped[float] = mapped_column(Float, nullable=False)
    algorithm_used: Mapped[str] = mapped_column(String(50), nullable=False)
    intent_type: Mapped[str | None] = mapped_column(String(50))
    narrative: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str | None] = mapped_column(String(128))
    updated_by: Mapped[str | None] = mapped_column(String(128))
    deleted_by: Mapped[str | None] = mapped_column(String(128))


class AnomalyAlert(Base):
    """Individual anomalous records from each detection session."""

    __tablename__ = "anomaly_alerts"
    __table_args__ = {"schema": _SCHEMA}

    id: Mapped[int | None] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(50), ForeignKey(f"{_SCHEMA}.anomaly_sessions.session_id"), nullable=False
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    explanation_text: Mapped[str | None] = mapped_column(Text)
    top_features: Mapped[list | None] = mapped_column(JSON)
    feature_values: Mapped[dict | None] = mapped_column(JSON)
    signal_sources: Mapped[list | None] = mapped_column(JSON)
    signal_breakdown: Mapped[dict | None] = mapped_column(JSON)
    is_dismissed: Mapped[bool] = mapped_column(default=False, nullable=False)
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    action_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    action_by: Mapped[str | None] = mapped_column(String(128))
    actor_timezone: Mapped[str | None] = mapped_column(String(64))
    actor_offset_minutes: Mapped[int | None] = mapped_column(Integer)


class RowEmbedding(Base):
    """384-dim sentence-transformer embeddings per row per session."""

    __tablename__ = "row_embeddings"
    __table_args__ = {"schema": _SCHEMA}

    id: Mapped[int | None] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(50), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    row_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_by: Mapped[str | None] = mapped_column(String(128))
