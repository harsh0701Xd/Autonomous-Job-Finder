-- docker/init.sql
-- Runs automatically on first Postgres startup via docker-entrypoint-initdb.d

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- LangGraph checkpoint tables are created automatically by the
-- PostgresSaver when the app first connects. This script handles
-- any additional app-level tables.

-- Session tracking table (application-level, separate from LangGraph)
CREATE TABLE IF NOT EXISTS sessions (
    session_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status          VARCHAR(50) NOT NULL DEFAULT 'created',
    location        TEXT,
    work_type       VARCHAR(20),
    jobs_found      INT DEFAULT 0,
    signals_found   INT DEFAULT 0
);

-- Index for status queries
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);

-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
