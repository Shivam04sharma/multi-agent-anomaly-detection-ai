"""Step 9 — Agent 5: Scoring Engine.

Fuses 3 independent anomaly signals into a single reliable score:

  final_score = (0.50 × statistical_score) + (0.30 × vector_score) + (0.20 × rule_flag)

  anomaly_flag = True  if final_score ≥ 0.65 (default)
  override     = True  if ANY single signal ≥ 0.90 (high-confidence)

Signal weights:
  0.50 — Statistical/ML model  (broadest data coverage)
  0.30 — Vector similarity      (captures semantic/contextual anomalies)
  0.20 — Rule flag              (domain-defined rules, highest precision)
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger()

_W_STAT = 0.50
_W_VEC = 0.30
_W_RULE = 0.20

_DEFAULT_THRESHOLD = 0.65
_OVERRIDE_THRESHOLD = 0.90  # any single signal this high → force flag


def _build_rule_scores(n_rows: int, vector_outlier_indices: list[int]) -> np.ndarray:
    """Rule signal array: 1.0 for vector-flagged rows, 0.0 otherwise.

    Can be extended with DB-configured rules in a future iteration.
    """
    scores = np.zeros(n_rows, dtype=float)
    for idx in vector_outlier_indices:
        if idx < n_rows:
            scores[idx] = 1.0
    return scores


def fuse_scores(
    statistical_scores: np.ndarray,
    vector_scores: list[float],
    vector_outlier_indices: list[int],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[dict]:
    """Fuse 3 signals and produce final anomaly determination per row.

    Adaptive threshold: when vector/rule signals are unavailable (all zero),
    threshold is scaled down proportionally so statistical signal alone
    can still flag anomalies correctly.

    Returns list of score dicts, one per row.
    """
    n_rows = len(statistical_scores)
    vec_arr = np.array(vector_scores) if vector_scores else np.zeros(n_rows)
    rule_arr = _build_rule_scores(n_rows, vector_outlier_indices)

    # Adaptive threshold — scale based on which signals are actually active
    has_vector = float(vec_arr.max()) > 0.0
    has_rule = float(rule_arr.max()) > 0.0
    active_weight = _W_STAT
    if has_vector:
        active_weight += _W_VEC
    if has_rule:
        active_weight += _W_RULE
    effective_threshold = threshold * active_weight

    logger.info(
        "scoring_threshold",
        configured=threshold,
        effective=round(effective_threshold, 4),
        has_vector=has_vector,
        has_rule=has_rule,
    )

    results = []
    for i in range(n_rows):
        stat = float(np.clip(statistical_scores[i], 0.0, 1.0))
        vec = float(np.clip(vec_arr[i], 0.0, 1.0)) if i < len(vec_arr) else 0.0
        rule = float(rule_arr[i])

        final_score = round(min((_W_STAT * stat) + (_W_VEC * vec) + (_W_RULE * rule), 1.0), 4)

        # High-confidence override: any single signal ≥ 0.90 → force flag
        override = any(s >= _OVERRIDE_THRESHOLD for s in [stat, vec, rule])
        anomaly_flag = override or (final_score >= effective_threshold)

        # Severity bucketing
        if final_score >= 0.80:
            severity = "high"
        elif final_score >= 0.55:
            severity = "medium"
        else:
            severity = "low"

        # Which signals actually fired (contributed meaningfully)
        signal_sources: list[str] = []
        if stat >= 0.60:
            signal_sources.append("statistical")
        if vec >= 0.60:
            signal_sources.append("vector")
        if rule >= 0.60:
            signal_sources.append("rule")

        results.append(
            {
                "row_index": i,
                "anomaly_score": final_score,
                "anomaly_flag": anomaly_flag,
                "severity": severity,
                "signal_breakdown": {
                    "stat": round(stat, 4),
                    "vector": round(vec, 4),
                    "rule": round(rule, 4),
                },
                "override_applied": override,
                "signal_sources": signal_sources,
            }
        )

    flagged = sum(1 for r in results if r["anomaly_flag"])
    logger.info(
        "scoring_complete",
        total=n_rows,
        flagged=flagged,
        threshold=threshold,
        override_count=sum(1 for r in results if r["override_applied"]),
    )
    return results
