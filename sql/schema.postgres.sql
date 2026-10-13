-- PostgreSQL schema: two tables. Same columns as schema.sqlite.sql.
-- db.py splits this file on the statement separator, so comments must not contain one.

CREATE TABLE IF NOT EXISTS alerts (
    id               BIGINT PRIMARY KEY,
    step             INTEGER NOT NULL,
    type             TEXT NOT NULL,
    amount           DOUBLE PRECISION NOT NULL,
    name_orig        TEXT NOT NULL,
    old_balance_orig DOUBLE PRECISION NOT NULL,
    new_balance_orig DOUBLE PRECISION NOT NULL,
    name_dest        TEXT NOT NULL,
    old_balance_dest DOUBLE PRECISION NOT NULL,
    new_balance_dest DOUBLE PRECISION NOT NULL,
    risk             DOUBLE PRECISION NOT NULL,
    model_risk       DOUBLE PRECISION NOT NULL,
    rule_risk        DOUBLE PRECISION NOT NULL,
    reasons          TEXT NOT NULL,
    is_fraud         SMALLINT,
    status           TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'approved', 'blocked')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alerts_status_risk ON alerts (status, risk DESC);

CREATE TABLE IF NOT EXISTS analyst_actions (
    id         BIGSERIAL PRIMARY KEY,
    alert_id   BIGINT NOT NULL REFERENCES alerts (id),
    analyst    TEXT NOT NULL CHECK (length(trim(analyst)) > 0),
    action     TEXT NOT NULL CHECK (action IN ('approve', 'block')),
    comment    TEXT NOT NULL CHECK (length(trim(comment)) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
