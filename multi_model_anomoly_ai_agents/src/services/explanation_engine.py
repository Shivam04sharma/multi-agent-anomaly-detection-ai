"""Step 10 — Agent 6: Explanation Engine.

Transforms raw anomaly scores into human-readable, non-technical explanations.

Two modes:
  - sync fallback  : rule-based plain English (no LLM, used when LLM unavailable)
  - async AI mode  : GPT-4o generates a friendly, jargon-free explanation that
                     any non-technical user can understand

Design principles:
  - Explanations are written for the end user, not the data scientist.
  - No sigma values, no "median", no ML terminology in the output.
  - Each explanation answers: "What is unusual?" and "Why does it matter?"
  - Batch AI explanations are generated in a single LLM call (one prompt, all
    flagged records) to minimise latency and cost.
"""

from __future__ import annotations

import json

import pandas as pd
import structlog

logger = structlog.get_logger()

_TOP_N_FEATURES = 3

# ── System prompt for per-record AI explanations ──────────────────────────────
_EXPLAIN_SYSTEM = (
    "You are a helpful assistant that explains data anomalies to non-technical users.\n\n"
    "You will receive a list of unusual records found in a dataset.\n"
    "For each record write ONE short, friendly sentence (max 25 words) that:\n"
    "- Explains what is unusual in plain English\n"
    '- Does NOT use technical terms like "sigma", "median", "score", '
    '"anomaly", "outlier", "statistical"\n'
    "- Reads like something a customer support agent would say\n"
    "- Focuses on the most surprising value\n\n"
    "Return a JSON array of strings, one explanation per record, in the same order as input.\n"
    'Example: ["This payment is unusually large compared to all other transactions.", '
    '"This login happened at an unusual hour from an unfamiliar location."]\n\n'
    "Return ONLY the JSON array. No markdown, no extra text."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _top_contributing_features(
    feature_df: pd.DataFrame,
    row_index: int,
    original_df: pd.DataFrame,
) -> tuple[list[str], dict]:
    """Find top N features with highest deviation from dataset median."""
    if feature_df.empty or row_index >= len(feature_df):
        return [], {}

    row = feature_df.iloc[row_index]
    medians = feature_df.median()
    stds = feature_df.std().replace(0, 1e-8)

    z_scores = ((row - medians) / stds).abs()
    top_cols = z_scores.nlargest(_TOP_N_FEATURES).index.tolist()

    feature_values: dict = {}
    for col in top_cols:
        if col in original_df.columns and row_index < len(original_df):
            feature_values[col] = (
                original_df.iloc[row_index][col].item()
                if hasattr(original_df.iloc[row_index][col], "item")
                else original_df.iloc[row_index][col]
            )
        else:
            feature_values[col] = round(float(row[col]), 4)

    return top_cols, feature_values


def _fallback_explanation(
    row_index: int,
    top_features: list[str],
    feature_values: dict,
) -> str:
    """Rule-based plain English fallback — used when LLM is unavailable.

    Avoids all technical language. Focuses on the most surprising value.
    """
    if not top_features:
        return "This record has an unusual combination of values."

    # Pick the single most surprising feature for the headline
    primary = top_features[0]
    primary_val = feature_values.get(primary, "unknown")

    # Clean up encoded column names (e.g. country_XX → country: XX)
    def _clean(col: str) -> str:
        if "_" in col:
            parts = col.rsplit("_", 1)
            return f"{parts[0].replace('_', ' ')}: {parts[1]}"
        return col.replace("_", " ")

    primary_label = _clean(primary)

    if len(top_features) > 1:
        secondary = top_features[1]
        secondary_val = feature_values.get(secondary, "unknown")
        secondary_label = _clean(secondary)
        return (
            f"This record stands out because {primary_label} is {primary_val}, "
            f"and {secondary_label} is {secondary_val}, which is very different from the rest."
        )

    return (
        f"This record stands out because {primary_label} is {primary_val}, "
        f"which is very different from all other records."
    )


def _build_user_prompt(flagged_records: list[dict]) -> str:
    """Build the user message for batch AI explanation generation."""
    records_summary = []
    for r in flagged_records:
        records_summary.append({
            "row": r["row_index"],
            "unusual_fields": r["feature_values"],
            "severity": r["severity"],
        })
    return json.dumps(records_summary, ensure_ascii=False)


# ── Async AI explanation generation ──────────────────────────────────────────

async def _generate_ai_explanations(
    flagged_records: list[dict],
    llm_client,
) -> list[str]:
    """Generate friendly AI explanations for all flagged records in one LLM call."""
    if not flagged_records:
        return []

    try:
        raw = await llm_client.complete(
            system=_EXPLAIN_SYSTEM,
            user=_build_user_prompt(flagged_records),
            temperature=0.4,
            max_tokens=60 * len(flagged_records),  # ~60 tokens per record
        )

        # Strip markdown fences if present
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        explanations: list[str] = json.loads(cleaned)

        if not isinstance(explanations, list) or len(explanations) != len(flagged_records):
            raise ValueError("LLM returned wrong number of explanations")

        return [str(e).strip() for e in explanations]

    except Exception as exc:
        logger.warning("ai_explanation_failed_fallback", error=str(exc))
        # Fall back to rule-based for all records
        return [
            _fallback_explanation(
                r["row_index"], r["top_features"], r["feature_values"]
            )
            for r in flagged_records
        ]


# ── Narrative ─────────────────────────────────────────────────────────────────

async def generate_narrative(
    anomalies: list[dict],
    total_rows: int,
    llm_client,
) -> str:
    """Generate a high-level plain-English summary for non-technical users."""
    if not anomalies:
        return f"Everything looks normal across all {total_rows} records checked."

    system = (
        "You are a helpful assistant summarising data findings for a non-technical audience. "
        "Write 2–3 friendly sentences that explain what was found and why it might matter. "
        "Do NOT use words like: anomaly, outlier, sigma, median, score, statistical, ML, model. "
        "Write as if explaining to a business manager. No markdown, no bullet points."
    )

    top = anomalies[:5]
    summary = "\n".join(
        f"- Row {a['row_index']}: severity={a['severity']}, "
        f"unusual fields={a.get('feature_values', {})}, "
        f"explanation={a.get('explanation_text', '')[:80]}"
        for a in top
    )
    user_msg = (
        f"We checked {total_rows} records and found {len(anomalies)} unusual ones.\n\n"
        f"Most notable findings:\n{summary}"
    )

    try:
        narrative = await llm_client.complete(
            system=system,
            user=user_msg,
            temperature=0.4,
            max_tokens=200,
        )
        return narrative.strip()
    except Exception as exc:
        logger.warning("narrative_generation_failed", error=str(exc))
        severity_counts: dict[str, int] = {}
        for a in anomalies:
            severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1
        parts = [f"{v} {k}-severity" for k, v in severity_counts.items()]
        return (
            f"Out of {total_rows} records checked, {len(anomalies)} looked unusual "
            f"({', '.join(parts)}). The most notable issue had a severity of "
            f"{anomalies[0]['severity']}."
        )


# ── Main explain function ─────────────────────────────────────────────────────

async def explain_async(
    scored_results: list[dict],
    original_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    llm_client,
) -> list[dict]:
    """Enrich each flagged record with AI-generated human-readable explanations.

    Uses a single batched LLM call for all flagged records — efficient and consistent.
    Non-flagged records get empty explanation fields.
    """
    # First pass — compute top features for all flagged records
    enriched_base: list[dict] = []
    flagged_with_features: list[dict] = []

    for record in scored_results:
        if not record["anomaly_flag"]:
            enriched_base.append({
                **record,
                "top_features": [],
                "feature_values": {},
                "explanation_text": "",
            })
            continue

        idx = record["row_index"]
        top_features, feature_values = _top_contributing_features(feature_df, idx, original_df)

        enriched = {
            **record,
            "top_features": top_features,
            "feature_values": {str(k): v for k, v in feature_values.items()},
            "explanation_text": "",  # filled below
        }
        enriched_base.append(enriched)
        flagged_with_features.append(enriched)

    # Second pass — generate AI explanations for all flagged records in one call
    if flagged_with_features:
        ai_explanations = await _generate_ai_explanations(flagged_with_features, llm_client)

        ai_idx = 0
        for record in enriched_base:
            if record["anomaly_flag"]:
                record["explanation_text"] = ai_explanations[ai_idx]
                ai_idx += 1

    flagged = sum(1 for r in enriched_base if r["anomaly_flag"])
    logger.info("explanations_generated", flagged=flagged, total=len(enriched_base), mode="ai")
    return enriched_base


def explain(
    scored_results: list[dict],
    original_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> list[dict]:
    """Sync fallback explain — rule-based only, no LLM.

    Used internally when async context is not available.
    """
    enriched = []
    for record in scored_results:
        if not record["anomaly_flag"]:
            enriched.append({
                **record,
                "top_features": [],
                "feature_values": {},
                "explanation_text": "",
            })
            continue

        idx = record["row_index"]
        top_features, feature_values = _top_contributing_features(feature_df, idx, original_df)
        explanation_text = _fallback_explanation(idx, top_features, feature_values)

        enriched.append({
            **record,
            "top_features": top_features,
            "feature_values": {str(k): v for k, v in feature_values.items()},
            "explanation_text": explanation_text,
        })

    flagged = sum(1 for r in enriched if r["anomaly_flag"])
    logger.info("explanations_generated", flagged=flagged, total=len(enriched), mode="fallback")
    return enriched
