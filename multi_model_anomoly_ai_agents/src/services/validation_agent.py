"""Step 2 — Agent 1: Data Validation.

Quality gate — if data fails critical checks, pipeline halts immediately.
Detects column types automatically (numeric, categorical, datetime).
"""

from __future__ import annotations

import pandas as pd
import structlog
from schemas.anomaly_schemas import ValidationResult

logger = structlog.get_logger()

MIN_ROWS = 10
_HIGH_NULL_COL_PCT = 70.0  # column with > 70% nulls → warning
_CRITICAL_NULL_COL_PCT = 95.0  # column with > 95% nulls → drop suggestion
_HIGH_NULL_OVERALL_PCT = 50.0  # if >50% of ALL cells are null → INVALID


def validate(df: pd.DataFrame) -> ValidationResult:
    """Validate dataset structure before any processing begins.

    Returns ValidationResult with dataset_status = 'VALID' or 'INVALID'.
    If INVALID, downstream agents must not run.
    """
    # 1. Minimum rows check
    if len(df) < MIN_ROWS:
        return ValidationResult(
            dataset_status="INVALID",
            numeric_columns=[],
            categorical_columns=[],
            datetime_columns=[],
            duplicate_count=0,
            missing_value_report={},
            reason=f"Dataset has only {len(df)} rows. Minimum {MIN_ROWS} required.",
        )

    # 2. Auto-detect column types
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == bool:
            numeric_columns.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)
        else:
            # Try to parse string column as datetime
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:  # 80%+ rows parse as datetime
                    datetime_columns.append(col)
                else:
                    categorical_columns.append(col)
            except Exception:
                categorical_columns.append(col)

    # 3. Must have at least numeric OR datetime columns
    # (datetime → numeric features extracted in feature_builder)
    if not numeric_columns and not datetime_columns:
        return ValidationResult(
            dataset_status="INVALID",
            numeric_columns=[],
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            duplicate_count=0,
            missing_value_report={},
            reason=(
                "No numeric or datetime columns found. "
                "Cannot perform anomaly detection on pure text data."
            ),
        )

    # 4. Duplicate row count
    duplicate_count = int(df.duplicated().sum())

    # 5. Missing value percentage per column
    missing_value_report = {col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns}

    # 6. Overall null rate — fail if dataset is mostly empty
    overall_null_pct = round(float(df.isna().mean().mean() * 100), 2)
    if overall_null_pct > _HIGH_NULL_OVERALL_PCT:
        return ValidationResult(
            dataset_status="INVALID",
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            duplicate_count=duplicate_count,
            missing_value_report=missing_value_report,
            reason=(
                f"Dataset has {overall_null_pct}% missing values overall "
                f"(threshold: {_HIGH_NULL_OVERALL_PCT}%). "
                "Too many nulls to perform reliable anomaly detection. "
                "Please clean or impute your data before uploading."
            ),
        )

    # 7. Per-column null warnings
    warnings: list[str] = []
    critical_cols = [
        col for col, pct in missing_value_report.items() if pct > _CRITICAL_NULL_COL_PCT
    ]
    high_null_cols = [
        col
        for col, pct in missing_value_report.items()
        if _HIGH_NULL_COL_PCT < pct <= _CRITICAL_NULL_COL_PCT
    ]
    if critical_cols:
        warnings.append(
            f"Columns with >95% missing values (consider dropping): {critical_cols}. "
            "These columns add minimal signal and may distort results."
        )
    if high_null_cols:
        warnings.append(
            f"Columns with >70% missing values (imputed with median): {high_null_cols}."
        )
    if duplicate_count > 0:
        dup_pct = round(duplicate_count / len(df) * 100, 1)
        if dup_pct > 20:
            warnings.append(
                f"{duplicate_count} duplicate rows ({dup_pct}% of dataset). "
                "High duplication may bias anomaly detection."
            )

    logger.info(
        "validation_passed",
        total_rows=len(df),
        numeric=len(numeric_columns),
        categorical=len(categorical_columns),
        datetime=len(datetime_columns),
        duplicates=duplicate_count,
        overall_null_pct=overall_null_pct,
        warnings=len(warnings),
    )

    return ValidationResult(
        dataset_status="VALID",
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        duplicate_count=duplicate_count,
        missing_value_report=missing_value_report,
        warnings=warnings,
    )
