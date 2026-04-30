-- ============================================================
-- Migration  : V3__add_missing_audit_columns
-- Env        : local
-- Description: Add missing audit columns to anomaly_sessions,
--              anomaly_alerts, row_embeddings
-- ============================================================

-- ── anomaly_sessions (Transactional) ─────────────────────────────────────────
ALTER TABLE {schema}.anomaly_sessions
    ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS deleted_at  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS created_by  VARCHAR(128),
    ADD COLUMN IF NOT EXISTS updated_by  VARCHAR(128);

-- ── anomaly_alerts (Audit/Event — only action_at) ────────────────────────────
ALTER TABLE {schema}.anomaly_alerts
    ADD COLUMN IF NOT EXISTS action_at TIMESTAMPTZ;

-- ── row_embeddings (Transactional) ───────────────────────────────────────────
ALTER TABLE {schema}.row_embeddings
    ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS deleted_at  TIMESTAMPTZ;
