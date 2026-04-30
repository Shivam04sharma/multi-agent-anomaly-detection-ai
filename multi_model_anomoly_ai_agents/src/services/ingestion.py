"""Step 1 — Data Ingestion Layer.

Accepts any tabular dataset — no hardcoded schema.
Converts DataPayload (columns + rows) or raw CSV bytes into a pandas DataFrame.
"""

from __future__ import annotations

import io

import pandas as pd
import structlog
from schemas.anomaly_schemas import DataPayload

logger = structlog.get_logger()


def ingest_payload(payload: DataPayload) -> pd.DataFrame:
    """Convert DataPayload into a pandas DataFrame.

    Works with any columns — fully dynamic, no hardcoded schema.
    """
    if not payload.columns:
        raise ValueError("No columns provided in data payload.")
    if not payload.rows:
        raise ValueError("No rows provided in data payload.")
    if len(payload.rows[0]) != len(payload.columns):
        raise ValueError(
            f"Column count mismatch: {len(payload.columns)} columns but "
            f"row has {len(payload.rows[0])} values."
        )

    df = pd.DataFrame(payload.rows, columns=payload.columns)

    # Auto-convert numeric-looking string columns
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.5:  # >50% values converted → keep
            df[col] = converted

    logger.info(
        "ingestion_complete",
        rows=len(df),
        columns=list(df.columns),
        dtypes={col: str(df[col].dtype) for col in df.columns},
    )
    return df


def ingest_csv(content: bytes) -> pd.DataFrame:
    """Convert raw CSV bytes into a pandas DataFrame."""
    df = pd.read_csv(io.BytesIO(content))

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.5:
            df[col] = converted

    logger.info("csv_ingestion_complete", rows=len(df), columns=list(df.columns))
    return df
