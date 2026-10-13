"""Shared test helpers."""
from __future__ import annotations

import pandas as pd

from fraud.db import SqliteDb


def txn(**over) -> dict:
    base = {"step": 100, "type": "TRANSFER", "amount": 5000.0, "name_orig": "C1",
            "old_balance_orig": 100000.0, "new_balance_orig": 95000.0, "name_dest": "C2",
            "old_balance_dest": 20000.0, "new_balance_dest": 25000.0}
    base.update(over)
    return base


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    """Raw rows plus history columns, with neutral history unless a row overrides it."""
    out = []
    for r in rows:
        d = {"vel_cnt": 0, "vel_amt": 0.0, "sender_prev_cnt": 0, "sender_prev_avg": None, "pair_prev_cnt": 0}
        d.update(txn())
        d.update(r)
        out.append(d)
    return pd.DataFrame(out)


def memory_db() -> SqliteDb:
    db = SqliteDb(":memory:")
    db.init_schema()
    return db


def add_alert(db, alert_id=1, risk=0.9, **over) -> int:
    row = {"id": alert_id, "risk": risk, "model_risk": risk, "rule_risk": 0.0,
           "reasons": '[{"source": "model", "code": "type", "text": "Transaction type is TRANSFER"}]',
           "is_fraud": 1, **txn(), "now": "2026-01-01T00:00:00+00:00"}
    row.update(over)
    db.query(
        "INSERT INTO alerts (id, step, type, amount, name_orig, old_balance_orig, new_balance_orig, "
        "name_dest, old_balance_dest, new_balance_dest, risk, model_risk, rule_risk, reasons, is_fraud, "
        "status, created_at) VALUES (:id, :step, :type, :amount, :name_orig, :old_balance_orig, "
        ":new_balance_orig, :name_dest, :old_balance_dest, :new_balance_dest, :risk, :model_risk, "
        ":rule_risk, :reasons, :is_fraud, 'open', :now)", row)
    return alert_id


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")
