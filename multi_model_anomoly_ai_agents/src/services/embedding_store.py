"""Save row embeddings to PostgreSQL for persistence.

Each analyze/upload run saves:
  - session_id   → ties embeddings to the analysis session
  - row_index    → which row this embedding belongs to
  - row_text     → the natural language sentence that was embedded
  - embedding    → 384-dim float array (sentence-transformers all-MiniLM-L6-v2)
"""

from __future__ import annotations

import asyncpg
import numpy as np
import structlog

logger = structlog.get_logger()


async def save_embeddings(
    pool: asyncpg.Pool | None,
    schema: str,
    session_id: str,
    row_texts: list[str],
    embeddings: np.ndarray,
) -> None:
    """Save all row embeddings for a session to PostgreSQL."""
    if pool is None:
        logger.debug("embedding_save_skipped", reason="DB not connected")
        return

    if embeddings is None or len(embeddings) == 0:
        return

    records = [
        (
            session_id,
            i,
            row_texts[i] if i < len(row_texts) else "",
            embeddings[i].tolist(),
        )
        for i in range(len(embeddings))
    ]

    async with pool.acquire() as conn:
        await conn.executemany(
            f"""
            INSERT INTO "{schema}".row_embeddings
                (session_id, row_index, row_text, embedding)
            VALUES ($1, $2, $3, $4)
            """,
            records,
        )

    logger.info(
        "embeddings_saved",
        session_id=session_id,
        rows=len(records),
        dim=embeddings.shape[1] if len(embeddings) > 0 else 0,
        schema=schema,
    )


async def get_embeddings(pool: asyncpg.Pool | None, schema: str, session_id: str) -> list[dict]:
    """Fetch saved embeddings for a session."""
    if pool is None:
        return []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT row_index, row_text, embedding
            FROM "{schema}".row_embeddings
            WHERE session_id = $1
            ORDER BY row_index ASC
            """,
            session_id,
        )
    return [dict(r) for r in rows]
