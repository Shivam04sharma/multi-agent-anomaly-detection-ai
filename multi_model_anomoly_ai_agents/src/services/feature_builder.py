"""Step 4 — Agent 2: Preprocessing & Feature Builder.

Transforms normalized dataset into ML-ready feature matrix.
Also generates natural language row descriptions for semantic embedding.

Handles ANY columns dynamically — no hardcoded schema.

Design principles (2024 MLOps standards):
- Identifier columns (user_id, uuid, keys) are detected and dropped before encoding
  to prevent spurious one-hot sparsity from polluting distance-based detectors (LOF/IF).
- Cardinality ratio ≥ 90% of row count → identifier → drop entirely.
- High-cardinality categoricals (≥ 20 unique) → target/frequency encoding instead of
  label encoding, which imposes false ordinal relationships.
- Near-zero variance columns dropped after encoding (not before) so encoded dummies
  that collapse to a single value are also caught.
- All thresholds are module-level constants — easy to tune without touching logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from schemas.anomaly_schemas import FeatureResult, ValidationResult

logger = structlog.get_logger()

# ── Thresholds ────────────────────────────────────────────────────────────────
_VARIANCE_THRESHOLD = 0.01       # near-zero variance → no discriminating signal → drop
_IDENTIFIER_RATIO = 0.90         # unique_values / n_rows ≥ this → identifier column → drop
_LOW_CARD_THRESHOLD = 20         # cardinality < this → one-hot; else → frequency encoding
_DATETIME_FEATURES = (           # time components extracted from datetime columns
    "hour", "weekday", "month", "is_weekend", "day_of_year"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_identifier(series: pd.Series) -> bool:
    """Return True if a column looks like a row identifier (user_id, uuid, etc.).

    Heuristic: if the ratio of unique values to total rows is ≥ _IDENTIFIER_RATIO
    the column carries no grouping signal — it's effectively a primary key.
    """
    if len(series) == 0:
        return False
    return series.nunique() / len(series) >= _IDENTIFIER_RATIO


def _frequency_encode(series: pd.Series) -> pd.Series:
    """Replace each category with its relative frequency in the dataset.

    Preserves ordinal signal (common categories score higher) without
    imposing arbitrary integer codes. Standard practice for high-cardinality
    categoricals in unsupervised anomaly detection.
    """
    freq_map = series.value_counts(normalize=True).to_dict()
    return series.map(freq_map).fillna(0.0)


def _extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Expand a datetime column into cyclic + binary time features."""
    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_hour"] = dt.dt.hour.fillna(0).astype(int)
    df[f"{col}_weekday"] = dt.dt.weekday.fillna(0).astype(int)
    df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
    df[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df[f"{col}_day_of_year"] = dt.dt.dayofyear.fillna(0).astype(int)
    return df.drop(columns=[col])


# ── Main ──────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    validation: ValidationResult,
) -> tuple[pd.DataFrame, FeatureResult]:
    """Build ML-ready numeric feature matrix from any tabular dataset.

    Pipeline order (matters):
      1. Drop identifier columns  ← NEW: prevents spurious LOF/IF signals
      2. Extract datetime features
      3. Encode remaining categoricals
      4. Select numeric columns only
      5. Drop near-zero variance columns

    Returns: (feature_df, FeatureResult)
    """
    df = df.copy()
    dropped_columns: list[str] = []
    encoding_map: dict[str, str] = {}

    # ── 1. Drop identifier columns ────────────────────────────────────────────
    # Must run BEFORE encoding — otherwise user_id becomes user_id_u1...uN
    # and each one-hot column appears exactly once, making every row look like
    # an outlier to LOF/Isolation Forest.
    identifier_cols = [
        col for col in validation.categorical_columns
        if col in df.columns and _is_identifier(df[col])
    ]
    if identifier_cols:
        df.drop(columns=identifier_cols, inplace=True)
        dropped_columns.extend(identifier_cols)
        logger.info("identifier_columns_dropped", columns=identifier_cols)

    # Remaining categoricals after identifier removal
    categorical_cols = [
        col for col in validation.categorical_columns
        if col in df.columns  # not already dropped
    ]

    # ── 2. Extract datetime features ──────────────────────────────────────────
    for col in validation.datetime_columns:
        if col not in df.columns:
            continue
        try:
            df = _extract_datetime_features(df, col)
            dropped_columns.append(col)
            logger.info("datetime_features_extracted", column=col)
        except Exception as exc:
            logger.warning("datetime_parse_failed", column=col, error=str(exc))
            df.drop(columns=[col], inplace=True)
            dropped_columns.append(col)

    # ── 3. Encode categorical columns ─────────────────────────────────────────
    for col in categorical_cols:
        if col not in df.columns:
            continue
        cardinality = df[col].nunique()

        if cardinality < _LOW_CARD_THRESHOLD:
            # One-hot for low cardinality (e.g. country: US/UK/XX → 3 columns)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoding_map[col] = f"one_hot(cardinality={cardinality})"
        else:
            # Frequency encoding for high cardinality — avoids false ordinal signal
            # of label encoding and avoids dimensionality explosion of one-hot
            df[col] = _frequency_encode(df[col])
            encoding_map[col] = f"frequency_encoding(cardinality={cardinality})"

    # ── 4. Keep only numeric columns ──────────────────────────────────────────
    feature_df = df.select_dtypes(include=[np.number]).copy()
    feature_df = feature_df.fillna(0)

    # ── 5. Drop near-zero variance columns ───────────────────────────────────
    # Runs after encoding so collapsed one-hot dummies are also caught
    low_var_cols = [
        col for col in feature_df.columns
        if feature_df[col].var() < _VARIANCE_THRESHOLD
    ]
    if low_var_cols:
        feature_df.drop(columns=low_var_cols, inplace=True)
        dropped_columns.extend(low_var_cols)
        logger.info("low_variance_dropped", columns=low_var_cols)

    feature_names = list(feature_df.columns)
    result = FeatureResult(
        feature_names=feature_names,
        dropped_columns=dropped_columns,
        encoding_map=encoding_map,
        row_count=len(feature_df),
    )

    logger.info(
        "features_built",
        features=len(feature_names),
        rows=len(feature_df),
        dropped=len(dropped_columns),
        identifiers_removed=len(identifier_cols),
    )
    return feature_df, result


def build_row_texts(original_df: pd.DataFrame) -> list[str]:
    """Convert each row into a natural language sentence for semantic embedding.

    Fully dynamic — works with any column names and values.
    Example: "revenue: 87000. month: Mar. region: North."
    """
    texts = []
    for _, row in original_df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        texts.append(". ".join(parts) + ".")
    return texts
