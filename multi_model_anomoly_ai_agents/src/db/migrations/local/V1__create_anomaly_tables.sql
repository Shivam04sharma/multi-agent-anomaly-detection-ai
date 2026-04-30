-- ============================================================
-- Migration  : V1__create_anomaly_tables
-- Env        : local
-- Description: Create schema and anomaly detection tables
-- Note       : {schema} is replaced at runtime from DB_SCHEMA env var
-- ============================================================

CREATE SCHEMA IF NOT EXISTS {schema};

-- Stores each anomaly detection run (one row per analyze call)
CREATE TABLE IF NOT EXISTS {schema}.anomaly_sessions (
    id             SERIAL          PRIMARY KEY,
    session_id     VARCHAR(50)     UNIQUE NOT NULL,  -- UUID per analyze call
    source_name    VARCHAR(255),                      -- CSV filename or source label
    total_rows     INTEGER         NOT NULL,
    anomalies_found INTEGER        NOT NULL,
    anomaly_rate   FLOAT           NOT NULL,
    algorithm_used VARCHAR(50)     NOT NULL,
    intent_type    VARCHAR(50),
    narrative      TEXT,
    created_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Stores individual anomalous records from each session
CREATE TABLE IF NOT EXISTS {schema}.anomaly_alerts (
    id               SERIAL          PRIMARY KEY,
    session_id       VARCHAR(50)     NOT NULL REFERENCES {schema}.anomaly_sessions(session_id),
    row_index        INTEGER         NOT NULL,
    anomaly_score    FLOAT           NOT NULL,
    severity         VARCHAR(20)     NOT NULL,   -- low / medium / high
    explanation_text TEXT,
    top_features     JSONB,                       -- ["revenue", "churn"]
    feature_values   JSONB,                       -- {"revenue": 87000, "churn": 0.45}
    signal_sources   JSONB,                       -- ["statistical", "vector"]
    signal_breakdown JSONB,                       -- {"stat": 0.7, "vector": 0.4, "rule": 0.0}
    is_dismissed     BOOLEAN         NOT NULL DEFAULT FALSE,
    dismissed_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by session
CREATE INDEX IF NOT EXISTS idx_alerts_session_id
    ON {schema}.anomaly_alerts (session_id);

-- Index for severity filter
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON {schema}.anomaly_alerts (severity);
