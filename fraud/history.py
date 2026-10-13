"""Sender history features computed in SQL with window functions.

The transactions are loaded into an in-memory SQLite table and one query computes, for
every row, what the sender and recipient did BEFORE that row. The same SQL runs unchanged
on PostgreSQL (checked on PostgreSQL 16, results identical).

Definitions (only earlier information is used, so there is no look-ahead):
  vel_cnt, vel_amt   count and total of the sender's transactions in the previous
                     VELOCITY_WINDOW_STEPS steps. Same-step rows are not counted: PaySim's
                     clock is one hour, so order inside an hour is unknown.
  sender_prev_cnt    number of earlier transactions by the sender
  sender_prev_avg    mean amount of those earlier transactions (NULL if none)
  pair_prev_cnt      earlier transactions from this sender to this recipient
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from .config import VELOCITY_WINDOW_STEPS

HISTORY_COLUMNS = ["vel_cnt", "vel_amt", "sender_prev_cnt", "sender_prev_avg", "pair_prev_cnt"]

BATCH_SQL = f"""
SELECT
    txn_id,
    COUNT(*) OVER w_vel                      AS vel_cnt,
    COALESCE(SUM(amount) OVER w_vel, 0.0)    AS vel_amt,
    COUNT(*) OVER w_sender                   AS sender_prev_cnt,
    AVG(amount) OVER w_sender                AS sender_prev_avg,
    COUNT(*) OVER w_pair                     AS pair_prev_cnt
FROM transactions
WINDOW
    w_vel AS (PARTITION BY name_orig ORDER BY step
              RANGE BETWEEN {VELOCITY_WINDOW_STEPS} PRECEDING AND 1 PRECEDING),
    w_sender AS (PARTITION BY name_orig ORDER BY step, txn_id
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w_pair AS (PARTITION BY name_orig, name_dest ORDER BY step, txn_id
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
"""


def batch_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return txn_id plus HISTORY_COLUMNS for every row of df."""
    con = sqlite3.connect(":memory:")
    try:
        df[["txn_id", "step", "name_orig", "name_dest", "amount"]].to_sql("transactions", con, index=False)
        out = pd.read_sql_query(BATCH_SQL, con)
    finally:
        con.close()
    for c in ("vel_cnt", "sender_prev_cnt", "pair_prev_cnt"):
        out[c] = out[c].astype(int)
    return out


def attach_history(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(batch_history(df), on="txn_id", how="left", validate="one_to_one")
