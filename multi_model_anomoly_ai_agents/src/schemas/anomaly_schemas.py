"""Pydantic schema contracts for the anomaly detection pipeline.

Every agent I/O is a typed Pydantic model — independently testable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class DataPayload(BaseModel):
    """Tabular dataset — any columns, any rows."""

    columns: list[str] = Field(..., description="Ordered list of column names.")
    rows: list[list[Any]] = Field(..., description="Dataset rows matching column order.")


class AnomalyAnalyzeRequest(BaseModel):
    """Full pipeline request — Steps 1-10."""

    data: DataPayload
    methods: list[str] = Field(
        default=["zscore", "isolation_forest"],
        description="Detection methods hint (used for logging; algorithm is auto-selected).",
    )
    narrate: bool = Field(default=True, description="Generate LLM narrative summary.")
    user_request: str | None = Field(
        default=None,
        description="Natural language intent, e.g. 'detect fraud in payments'.",
    )
    sensitivity: float | None = Field(
        default=None,
        description="Override contamination: 0.05 (low) | 0.10 (medium) | 0.20 (high).",
    )


class AnomalyPredictRequest(BaseModel):
    """Single-record real-time prediction request."""

    record: dict[str, Any] = Field(..., description="Single row as column→value dict.")
    reference_data: DataPayload | None = Field(
        default=None,
        description="Reference dataset for context (used to build scaler baseline).",
    )
    narrate: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Pipeline intermediate contracts
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Output of Agent 1 — Data Validation (Step 2)."""

    dataset_status: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    duplicate_count: int
    missing_value_report: dict[str, float]
    reason: str | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Data quality warnings (high nulls, high duplicates, etc.).",
    )


class NormalizationResult(BaseModel):
    """Output of normalization layer (Step 3)."""

    scaler_map: dict[str, str]
    imputation_log: dict[str, str]
    transform_log: dict[str, str]
    distribution_report: dict[str, float]


class FeatureResult(BaseModel):
    """Output of Agent 2 — Feature Builder (Step 4)."""

    feature_names: list[str]
    dropped_columns: list[str]
    encoding_map: dict[str, str]
    row_count: int


class IntentConfig(BaseModel):
    """Output of Agent 3 — Prompt Builder (Step 5)."""

    anomaly_type: str = Field(
        description="fraud | security | operational | pattern | custom | default"
    )
    focus_columns: list[str] = Field(description="Columns most relevant to the objective.")
    sensitivity_level: float | str = Field(
        description=(
            "'auto' = algorithm decides (default) | "
            "0.05 = flag only extreme anomalies | "
            "0.10 = balanced | 0.20 = high recall"
        )
    )
    intent_source: str = Field(description="user_input | default_fallback | ambiguity_default")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class AnomalyDetected(BaseModel):
    """A single anomalous record with full explanation."""

    row_index: int
    anomaly_score: float = Field(description="Weighted final score 0.0–1.0.")
    anomaly_flag: bool
    severity: str = Field(description="low | medium | high")
    signal_breakdown: dict[str, float] = Field(
        description="Individual signal scores: {'stat': 0.7, 'vector': 0.4, 'rule': 1.0}"
    )
    override_applied: bool = Field(description="True if high-confidence override fired.")
    top_features: list[str]
    feature_values: dict[str, Any]
    explanation_text: str
    signal_sources: list[str] = Field(
        description="Which signals fired: statistical / vector / rule"
    )


class AnomalyAnalyzeResponse(BaseModel):
    """Full pipeline response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    total_rows: int
    anomalies_found: int
    anomaly_rate: float
    anomalies: list[AnomalyDetected]
    narrative: str | None = None
    algorithm_used: str
    model_used: str
    validation: ValidationResult
    intent: IntentConfig
    processing_notes: list[str] = Field(default_factory=list)


class AnomalyPredictResponse(BaseModel):
    """Single-record prediction response."""

    model_config = ConfigDict(protected_namespaces=())

    anomaly: bool
    anomaly_score: float
    severity: str
    explanation: str
    signal_breakdown: dict[str, float]
    model_used: str


# ---------------------------------------------------------------------------
# Sessions list models
# ---------------------------------------------------------------------------


class AlertPreview(BaseModel):
    """Top anomalous record preview inside a session summary."""

    row_index: int
    anomaly_score: float
    severity: str
    explanation_text: str
    top_features: list[str]


class SessionSummary(BaseModel):
    """One session with aggregated stats and top alert previews."""

    session_id: str
    source_name: str | None
    total_rows: int
    anomalies_found: int
    anomaly_rate: float
    algorithm_used: str
    intent_type: str | None
    narrative: str | None
    created_at: datetime
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    top_alerts_preview: list[AlertPreview]


class SessionsListResponse(BaseModel):
    """Paginated list of all sessions with summaries."""

    total: int
    limit: int
    offset: int
    sessions: list[SessionSummary]
