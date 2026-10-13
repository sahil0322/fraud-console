-- SQLite schema for local use: two tables. Same columns as schema.postgres.sql.

CREATE TABLE IF NOT EXISTS alerts (
    id               INTEGER PRIMARY KEY,
    step             INTEGER NOT NULL,
    type             TEXT NOT NULL,
    amount           REAL NOT NULL,
    name_orig        TEXT NOT NULL,
    old_balance_orig REAL NOT NULL,
    new_balance_orig REAL NOT NULL,
    name_dest        TEXT NOT NULL,
    old_balance_dest REAL NOT NULL,
    new_balance_dest REAL NOT NULL,
    risk             REAL NOT NULL,
    model_risk       REAL NOT NULL,
    rule_risk        REAL NOT NULL,
    reasons          TEXT NOT NULL,
    is_fraud         INTEGER,
    status           TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'approved', 'blocked')),
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_status_risk ON alerts (status, risk DESC);

CREATE TABLE IF NOT EXISTS analyst_actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id   INTEGER NOT NULL REFERENCES alerts (id),
    analyst    TEXT NOT NULL CHECK (length(trim(analyst)) > 0),
    action     TEXT NOT NULL CHECK (action IN ('approve', 'block')),
    comment    TEXT NOT NULL CHECK (length(trim(comment)) > 0),
    created_at TEXT NOT NULL
);
