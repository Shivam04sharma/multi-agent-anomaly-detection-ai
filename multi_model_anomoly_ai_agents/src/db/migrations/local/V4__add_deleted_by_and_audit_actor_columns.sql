-- ============================================================
-- Migration  : V4__add_deleted_by_and_audit_actor_columns
-- Env        : local
-- Description: Add deleted_by to transactional tables,
--              action_by + actor_timezone + actor_offset_minutes to audit tables
-- ============================================================

-- ── anomaly_sessions: add deleted_by ─────────────────────────────────────────
ALTER TABLE {schema}.anomaly_sessions
    ADD COLUMN IF NOT EXISTS deleted_by VARCHAR(128);

-- ── anomaly_alerts: add action_by, actor_timezone, actor_offset_minutes ───────
ALTER TABLE {schema}.anomaly_alerts
    ADD COLUMN IF NOT EXISTS action_by            VARCHAR(128),
    ADD COLUMN IF NOT EXISTS actor_timezone       VARCHAR(64),
    ADD COLUMN IF NOT EXISTS actor_offset_minutes INTEGER;

-- ── row_embeddings: add deleted_by ───────────────────────────────────────────
ALTER TABLE {schema}.row_embeddings
    ADD COLUMN IF NOT EXISTS deleted_by VARCHAR(128);
