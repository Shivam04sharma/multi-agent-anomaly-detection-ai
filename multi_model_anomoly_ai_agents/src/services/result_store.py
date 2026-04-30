"""Save anomaly detection results to PostgreSQL.

Saves:
  anomaly_sessions — one row per analyze/upload call
  anomaly_alerts   — one row per flagged anomalous record
"""

from __future__ import annotations

import json

import asyncpg
import structlog

logger = structlog.get_logger()


async def save_session(
    pool: asyncpg.Pool | None,
    schema: str,
    session_id: str,
    source_name: str,
    total_rows: int,
    anomalies_found: int,
    anomaly_rate: float,
    algorithm_used: str,
    intent_type: str,
    narrative: str | None,
) -> None:
    if pool is None:
        logger.debug("db_save_skipped", reason="DB not connected")
        return
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO "{schema}".anomaly_sessions
                (session_id, source_name, total_rows, anomalies_found,
                 anomaly_rate, algorithm_used, intent_type, narrative)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id,
            source_name,
            total_rows,
            anomalies_found,
            anomaly_rate,
            algorithm_used,
            intent_type,
            narrative,
        )
    logger.info("session_saved", session_id=session_id, anomalies=anomalies_found)


async def save_alerts(
    pool: asyncpg.Pool | None,
    schema: str,
    session_id: str,
    anomalies: list[dict],
) -> None:
    if not anomalies:
        return
    if pool is None:
        logger.debug("db_save_skipped", reason="DB not connected")
        return
    async with pool.acquire() as conn:
        await conn.executemany(
            f"""
            INSERT INTO "{schema}".anomaly_alerts
                (session_id, row_index, anomaly_score, severity,
                 explanation_text, top_features, feature_values,
                 signal_sources, signal_breakdown)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            [
                (
                    session_id,
                    a["row_index"],
                    a["anomaly_score"],
                    a["severity"],
                    a.get("explanation_text", ""),
                    json.dumps(a.get("top_features", [])),
                    json.dumps(a.get("feature_values", {})),
                    json.dumps(a.get("signal_sources", [])),
                    json.dumps(a.get("signal_breakdown", {})),
                )
                for a in anomalies
            ],
        )
    logger.info("alerts_saved", session_id=session_id, count=len(anomalies))


async def get_session_alerts(pool: asyncpg.Pool | None, schema: str, session_id: str) -> list[dict]:
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT * FROM "{schema}".anomaly_alerts
            WHERE session_id = $1
            ORDER BY anomaly_score DESC
            """,
            session_id,
        )
    return [dict(r) for r in rows]


async def get_all_sessions(
    pool: asyncpg.Pool | None,
    schema: str,
    limit: int = 50,
    offset: int = 0,
    intent_type: str | None = None,
    min_anomaly_rate: float | None = None,
) -> list[dict]:
    """Fetch all sessions with summary stats and top alert preview."""
    if pool is None:
        return []

    # Build optional filters
    filters = []
    params: list = []
    p = 1

    if intent_type:
        filters.append(f"s.intent_type = ${p}")
        params.append(intent_type)
        p += 1

    if min_anomaly_rate is not None:
        filters.append(f"s.anomaly_rate >= ${p}")
        params.append(min_anomaly_rate)
        p += 1

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    params.extend([limit, offset])

    async with pool.acquire() as conn:
        # Main sessions query with severity breakdown via lateral join
        sessions = await conn.fetch(
            f"""
            SELECT
                s.session_id,
                s.source_name,
                s.total_rows,
                s.anomalies_found,
                s.anomaly_rate,
                s.algorithm_used,
                s.intent_type,
                s.narrative,
                s.created_at,
                COALESCE(sev.high_count, 0)   AS high_severity_count,
                COALESCE(sev.medium_count, 0) AS medium_severity_count,
                COALESCE(sev.low_count, 0)    AS low_severity_count
            FROM "{schema}".anomaly_sessions s
            LEFT JOIN LATERAL (
                SELECT
                    COUNT(*) FILTER (WHERE severity = 'high')   AS high_count,
                    COUNT(*) FILTER (WHERE severity = 'medium') AS medium_count,
                    COUNT(*) FILTER (WHERE severity = 'low')    AS low_count
                FROM "{schema}".anomaly_alerts
                WHERE session_id = s.session_id
            ) sev ON true
            {where_clause}
            ORDER BY s.created_at DESC
            LIMIT ${p} OFFSET ${p + 1}
            """,
            *params,
        )

        # For each session fetch top 3 alerts as preview
        result = []
        for s in sessions:
            top_alerts = await conn.fetch(
                f"""
                SELECT row_index, anomaly_score, severity, explanation_text, top_features
                FROM "{schema}".anomaly_alerts
                WHERE session_id = $1
                ORDER BY anomaly_score DESC
                LIMIT 3
                """,
                s["session_id"],
            )

            result.append(
                {
                    **dict(s),
                    "top_alerts_preview": [dict(a) for a in top_alerts],
                }
            )

    return result


async def get_sessions_count(
    pool: asyncpg.Pool | None,
    schema: str,
    intent_type: str | None = None,
    min_anomaly_rate: float | None = None,
) -> int:
    """Total session count for pagination."""
    if pool is None:
        return 0

    filters = []
    params: list = []
    p = 1

    if intent_type:
        filters.append(f"intent_type = ${p}")
        params.append(intent_type)
        p += 1

    if min_anomaly_rate is not None:
        filters.append(f"anomaly_rate >= ${p}")
        params.append(min_anomaly_rate)
        p += 1

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}".anomaly_sessions {where_clause}',
            *params,
        )
    return count or 0
