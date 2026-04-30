"""Step 3 — Data Normalization.

Outlier-resistant preprocessing:
- KNN imputation (k=5) for missing numeric values
- RobustScaler default (uses median + IQR — unaffected by outliers)
- Log-transform for skewed columns (skewness > 1.0) before scaling
- StandardScaler only for confirmed normal distribution (skewness < 0.5)
- MinMaxScaler is NOT used (distorts anomaly signal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from schemas.anomaly_schemas import NormalizationResult, ValidationResult

logger = structlog.get_logger()

_SKEW_HIGH = 1.0  # apply log-transform + RobustScaler
_SKEW_NORMAL = 0.5  # apply StandardScaler


def normalize(
    df: pd.DataFrame,
    validation: ValidationResult,
) -> tuple[pd.DataFrame, NormalizationResult]:
    """Standardize dataset for ML algorithms.

    Returns: (normalized_df, NormalizationResult)
    """
    df = df.copy()
    scaler_map: dict[str, str] = {}
    imputation_log: dict[str, str] = {}
    transform_log: dict[str, str] = {}
    distribution_report: dict[str, float] = {}

    numeric_cols = list(validation.numeric_columns)
    categorical_cols = list(validation.categorical_columns)

    # 1. Boolean → int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            if col not in numeric_cols:
                numeric_cols.append(col)

    # 2. KNN imputation for missing numeric values
    if numeric_cols:
        from sklearn.impute import KNNImputer

        cols_with_missing = [c for c in numeric_cols if df[c].isna().any()]
        if cols_with_missing:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            for col in cols_with_missing:
                imputation_log[col] = "knn_imputer_k5"
            logger.info("knn_imputation_applied", columns=cols_with_missing)

    # 3. Categorical — fill with most frequent value or 'Unknown'
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_vals = df[col].mode()
            fill_val = str(mode_vals.iloc[0]) if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            imputation_log[col] = f"mode_fill:{fill_val}"

    # 4. Scale numeric columns based on distribution
    for col in numeric_cols:
        if col not in df.columns:
            continue

        skewness = float(df[col].skew())
        distribution_report[col] = round(skewness, 4)

        if skewness > _SKEW_HIGH:
            # Log-transform (clip negatives to 0) then RobustScaler
            from sklearn.preprocessing import RobustScaler

            df[col] = np.log1p(df[col].clip(lower=0))
            transform_log[col] = "log1p"
            scaler = RobustScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
            scaler_map[col] = "log+RobustScaler"

        elif abs(skewness) < _SKEW_NORMAL:
            # StandardScaler for confirmed normal distribution
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
            scaler_map[col] = "StandardScaler"

        else:
            # Default: RobustScaler — resistant to outlier influence
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
            scaler_map[col] = "RobustScaler"

    logger.info(
        "normalization_complete",
        columns_scaled=len(scaler_map),
        log_transformed=len(transform_log),
        imputed=len(imputation_log),
    )

    result = NormalizationResult(
        scaler_map=scaler_map,
        imputation_log=imputation_log,
        transform_log=transform_log,
        distribution_report=distribution_report,
    )
    return df, result
