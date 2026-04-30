-- ============================================================
-- Migration  : V2__create_embeddings_table
-- Env        : local
-- Description: Store row-level embeddings per session
-- Note       : {schema} is replaced at runtime from DB_SCHEMA env var
-- ============================================================

-- Stores 384-dim sentence-transformer embeddings per row per session
CREATE TABLE IF NOT EXISTS {schema}.row_embeddings (
    id           SERIAL          PRIMARY KEY,
    session_id   VARCHAR(50)     NOT NULL,
    row_index    INTEGER         NOT NULL,
    row_text     TEXT            NOT NULL,   -- the natural language sentence that was embedded
    embedding    FLOAT8[]        NOT NULL,   -- 384-dim vector as PostgreSQL array
    created_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_session_id
    ON {schema}.row_embeddings (session_id);
