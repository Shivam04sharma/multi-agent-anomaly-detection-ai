"""Step 8 — Agent 4: Detection Strategy Layer.

Auto-selects the most appropriate anomaly detection algorithm using
4 dimensions: data type, label availability, anomaly rate, latency.

Available algorithms:
- isolation_forest  : tabular numeric, unsupervised, batch (default)
- statistical_zscore: real-time, latency < 50ms
- lof               : small datasets (< 1000 rows)
- statistical_mad   : robust alternative to Z-score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from schemas.anomaly_schemas import IntentConfig

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Individual detection methods
# ---------------------------------------------------------------------------


def _zscore_scores(feature_df: pd.DataFrame) -> np.ndarray:
    """Z-score statistical detection. Returns per-row anomaly score (0–1)."""
    z = np.abs((feature_df - feature_df.mean()) / (feature_df.std() + 1e-8))
    row_max = z.max(axis=1).values
    max_val = row_max.max() if row_max.max() > 0 else 1.0
    return np.clip(row_max / max_val, 0.0, 1.0)


def _mad_scores(feature_df: pd.DataFrame) -> np.ndarray:
    """Median Absolute Deviation — robust alternative to Z-score."""
    median = feature_df.median()
    mad = (feature_df - median).abs().median()
    mad_safe = mad.replace(0, 1e-8)
    modified_z = 0.6745 * (feature_df - median) / mad_safe
    row_max = modified_z.abs().max(axis=1).values
    max_val = row_max.max() if row_max.max() > 0 else 1.0
    return np.clip(row_max / max_val, 0.0, 1.0)


def _isolation_forest_scores(
    feature_df: pd.DataFrame,
    contamination: float | str,
) -> tuple[np.ndarray, object]:
    """Isolation Forest — best for tabular numeric unsupervised detection.

    contamination="auto" → sklearn decides threshold from data distribution.
    contamination=float  → user/LLM-specified rate (0.05–0.50).
    Rows predicted as anomaly (-1) get score boosted to min 0.70
    so they always cross the scoring threshold.
    """
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        contamination=contamination,  # "auto" or float — both accepted by sklearn
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    predictions = model.fit_predict(feature_df)  # -1 anomaly, 1 normal
    decision = model.decision_function(feature_df)

    # Normalize decision score to 0–1
    min_d, max_d = decision.min(), decision.max()
    if max_d - min_d > 0:
        normalized = 1.0 - (decision - min_d) / (max_d - min_d)
    else:
        normalized = np.zeros(len(decision))

    # IF-flagged rows get minimum score of 0.70 — guarantees they pass threshold
    normalized[predictions == -1] = np.maximum(normalized[predictions == -1], 0.70)

    return np.clip(normalized, 0.0, 1.0), model


def _lof_scores(
    feature_df: pd.DataFrame,
    contamination: float | str,
) -> np.ndarray:
    """Local Outlier Factor — best for small datasets (< 1000 rows)."""
    from sklearn.neighbors import LocalOutlierFactor

    n_neighbors = min(20, len(feature_df) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof.fit_predict(feature_df)

    # negative_outlier_factor_: more negative = more anomalous
    scores = lof.negative_outlier_factor_
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s > 0:
        normalized = (scores - max_s) / (min_s - max_s)
    else:
        normalized = np.zeros(len(scores))

    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Algorithm selection + execution
# ---------------------------------------------------------------------------


def select_and_detect(
    feature_df: pd.DataFrame,
    intent: IntentConfig,
    latency_mode: str = "batch",
) -> tuple[np.ndarray, str, str, object | None]:
    """Auto-select algorithm and run detection.

    Selection matrix (4 dimensions):
    - latency_mode=realtime + any size   → Z-score (< 50ms)
    - n_rows < 1000                      → LOF
    - n_cols > 40                        → Isolation Forest
    - default                            → Isolation Forest

    Returns:
        statistical_scores : per-row score array (0–1)
        algorithm_name     : which algorithm was used
        selection_reason   : human-readable explanation
        model_object       : fitted model (or None for statistical)
    """
    n_rows, n_cols = feature_df.shape
    # "auto" → sklearn decides from data distribution (recommended default)
    # float  → user/LLM explicitly set a contamination rate
    contamination: float | str = (
        intent.sensitivity_level
        if intent.sensitivity_level == "auto"
        else float(intent.sensitivity_level)
    )

    # Focus on relevant columns if intent specifies them
    focus_cols = [c for c in intent.focus_columns if c in feature_df.columns]
    working_df = feature_df[focus_cols] if focus_cols else feature_df

    # Guard: need at least 2 columns for multivariate methods
    if working_df.shape[1] < 2:
        working_df = feature_df

    logger.info(
        "algorithm_selection",
        rows=n_rows,
        cols=n_cols,
        latency_mode=latency_mode,
        contamination=contamination,
    )

    # --- Selection logic ---
    if latency_mode == "realtime":
        scores = _zscore_scores(working_df)
        return scores, "statistical_zscore", "Real-time mode — Z-score for latency < 50ms.", None

    # Very small datasets (< 50 rows): LOF n_neighbors collapses to dataset size,
    # making distance geometry unreliable. Z-score is more robust here because
    # it operates per-column independently and is not affected by neighborhood size.
    if n_rows < 50:
        scores = _zscore_scores(working_df)
        return (
            scores,
            "statistical_zscore",
            (
                f"Very small dataset ({n_rows} rows) — "
                "Z-score selected (LOF unreliable below 50 rows)."
            ),
            None,
        )

    # Medium datasets (50–999 rows): LOF works well, neighborhood is meaningful
    if n_rows < 1000:
        scores = _lof_scores(working_df, contamination)
        return scores, "lof", f"Small dataset ({n_rows} rows) — LOF selected.", None

    # Default: Isolation Forest (handles any dimension, best general-purpose)
    scores, model = _isolation_forest_scores(working_df, contamination)
    reason = (
        f"Tabular numeric ({n_rows} rows, {n_cols} features) — "
        "Isolation Forest selected (best general-purpose unsupervised detector)."
    )
    return scores, "isolation_forest", reason, model
