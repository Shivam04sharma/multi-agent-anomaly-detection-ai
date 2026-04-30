"""Step 5 — Agent 3: Requirement & Prompt Builder.

Translates user's natural language request into a precise detection configuration
using GPT-4o intent parsing. Falls back to safe defaults if no request provided.

Design principles (2024 LLM-ops standards):
- Sensitivity is ALWAYS "auto" unless the user explicitly overrides it.
  The LLM is not trusted to set contamination rates — it has no knowledge of
  the actual data distribution. Algorithm auto-selection is more reliable.
- focus_columns passed to the LLM are pre-filtered to exclude one-hot expanded
  columns (e.g. country_US, country_UK) — the LLM should reason about original
  column names, not encoded artifacts.
- Structured output via JSON mode with strict field validation.
- Fallback chain: user_input → ambiguity_default → default_fallback
  Each level is logged with a distinct intent_source for observability.
- System prompt uses few-shot examples for more reliable JSON extraction.
"""

from __future__ import annotations

import json
import re

import structlog
from schemas.anomaly_schemas import IntentConfig

logger = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

_VALID_ANOMALY_TYPES = frozenset(
    {"fraud", "security", "operational", "pattern", "custom", "default"}
)

# Regex to detect one-hot expanded column names (e.g. country_US, status_active)
# These are encoding artifacts — not meaningful to the LLM
_ENCODED_COL_PATTERN = re.compile(r"^.+_[A-Za-z0-9]+$")

_SYSTEM_PROMPT = """You are an anomaly detection configuration assistant.

Given a user's natural language request and a list of available dataset columns,
return a JSON object with EXACTLY these fields — no extra keys, no markdown:

{
  "anomaly_type": "fraud" | "security" | "operational" | "pattern" | "custom",
  "focus_columns": ["col1", "col2", ...]
}

Rules:
- anomaly_type must be one of the five values above.
  fraud       → financial transactions, payments, amounts, accounts
  security    → access logs, IPs, login attempts, permissions
  operational → latency, errors, throughput, system metrics
  pattern     → time-series, sequences, behavioral patterns
  custom      → anything else
- focus_columns must be a subset of the provided available_columns list.
  Pick only columns directly relevant to the objective.
  Prefer numeric columns. Exclude identifier-like columns (ids, keys, names).
  If unsure, return all numeric columns.
- Do NOT include a sensitivity_level field — the algorithm decides this automatically.
- Return ONLY valid JSON. No explanation, no markdown fences, no extra text.

Examples:

User: "detect fraud in payment transactions"
Columns: ["amount", "merchant_id", "country", "duration_sec", "is_weekend"]
Response: {"anomaly_type": "fraud", "focus_columns": ["amount", "duration_sec", "is_weekend"]}

User: "find unusual login patterns"
Columns: ["ip_address", "login_hour", "failed_attempts", "country", "user_agent"]
Response: {"anomaly_type": "security", "focus_columns": ["login_hour", "failed_attempts"]}

User: "detect latency spikes in API calls"
Columns: ["endpoint", "latency_ms", "status_code", "tokens_used", "timestamp_hour"]
Response: {"anomaly_type": "operational", "focus_columns": ["latency_ms", "status_code"]}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_original_columns(columns: list[str]) -> list[str]:
    """Return only original (non-encoded) column names for LLM context.

    Strips one-hot artifacts like country_US, status_active so the LLM
    reasons about the original schema, not encoding internals.
    Keeps columns that look like raw feature names (no underscore-suffix pattern
    that matches an encoded dummy).
    """
    # Heuristic: if a column name looks like {original}_{value} AND
    # there are multiple columns sharing the same prefix → it's encoded
    prefixes: dict[str, int] = {}
    for col in columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            prefixes[parts[0]] = prefixes.get(parts[0], 0) + 1

    filtered = []
    for col in columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and prefixes.get(parts[0], 0) > 1:
            # Looks like an encoded dummy — skip
            continue
        filtered.append(col)

    return filtered if filtered else columns


def _safe_anomaly_type(value: str | None) -> str:
    """Validate and normalise anomaly_type from LLM response."""
    if not value:
        return "custom"
    cleaned = str(value).strip().lower()
    return cleaned if cleaned in _VALID_ANOMALY_TYPES else "custom"


# ── Main ──────────────────────────────────────────────────────────────────────

async def parse_intent(
    user_request: str | None,
    available_columns: list[str],
    llm_client,
    sensitivity_override: float | None = None,
) -> IntentConfig:
    """Parse user intent into a detection configuration.

    Sensitivity is always "auto" unless the caller explicitly overrides it.
    The LLM is not asked for sensitivity — it has no knowledge of data distribution.

    Fallback chain:
      1. user_input       — LLM parsed successfully
      2. ambiguity_default — LLM failed; use numeric-only columns
      3. default_fallback  — no user_request provided at all
    """
    # Sensitivity: explicit override wins; otherwise always "auto"
    sensitivity: float | str = (
        float(sensitivity_override) if sensitivity_override is not None else "auto"
    )

    # ── No request → safe default ─────────────────────────────────────────────
    if not user_request or not user_request.strip():
        logger.info("intent_default_fallback", reason="no_user_request")
        return IntentConfig(
            anomaly_type="default",
            focus_columns=available_columns,
            sensitivity_level=sensitivity,
            intent_source="default_fallback",
        )

    # Strip encoding artifacts before sending to LLM
    llm_columns = _filter_original_columns(available_columns)

    user_msg = json.dumps(
        {
            "user_request": user_request.strip(),
            "available_columns": llm_columns,
        },
        ensure_ascii=False,
    )

    try:
        result = await llm_client.complete_json(
            system=_SYSTEM_PROMPT,
            user=user_msg,
            temperature=0.0,   # deterministic — intent parsing is not creative
            max_tokens=256,
        )

        # Validate focus_columns — must be real columns from available_columns
        raw_cols: list[str] = result.get("focus_columns", [])
        valid_cols = [c for c in raw_cols if c in available_columns]

        # If LLM returned encoded column names, map them back
        if not valid_cols:
            valid_cols = [c for c in raw_cols if c in llm_columns]

        # Final fallback: use all available columns if LLM returned nothing useful
        if not valid_cols:
            valid_cols = available_columns
            logger.warning("intent_focus_columns_empty_fallback", raw=raw_cols)

        config = IntentConfig(
            anomaly_type=_safe_anomaly_type(result.get("anomaly_type")),
            focus_columns=valid_cols,
            sensitivity_level=sensitivity,   # never from LLM
            intent_source="user_input",
        )

        logger.info(
            "intent_parsed",
            anomaly_type=config.anomaly_type,
            focus_columns=valid_cols,
            sensitivity=config.sensitivity_level,
        )
        return config

    except Exception as exc:
        logger.warning("intent_parse_failed", error=str(exc))

        # Ambiguity fallback — use only numeric-looking columns (no encoded dummies)
        numeric_fallback = _filter_original_columns(available_columns)

        return IntentConfig(
            anomaly_type="custom",
            focus_columns=numeric_fallback,
            sensitivity_level=sensitivity,
            intent_source="ambiguity_default",
        )
