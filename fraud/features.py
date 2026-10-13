"""Feature engineering. One function serves training and live scoring.

Input frame needs the raw transaction columns plus HISTORY_COLUMNS (see history.py).
Output frame holds exactly NUMERIC_FEATURES + CATEGORICAL_FEATURES, plus a few
helper columns the rule engine reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DRAIN_FRACTION, OUTFLOW_TYPES

RAW_COLUMNS = [
    "step", "type", "amount", "name_orig", "old_balance_orig", "new_balance_orig",
    "name_dest", "old_balance_dest", "new_balance_dest",
]
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "amount_log",
    "hour_of_day",
    "balance_frac",         # amount as a share of the sender's balance, capped at 2
    "drains_account",       # 1 if an outflow takes at least DRAIN_FRACTION of the balance
    "orig_balance_zero",    # 1 if the sender balance was zero before the transaction
    "err_orig_signed_log",  # signed log of (expected new sender balance - recorded)
    "err_dest_signed_log",  # signed log of (expected new recipient balance - recorded)
    "dest_old_zero",        # 1 if the recipient balance was zero before (new-looking account)
    "first_time_beneficiary",
    "vel_cnt",
    "vel_amt_log",
    "amount_vs_avg_log",    # log of amount relative to the sender's own earlier average
    "has_history",          # 1 if the sender has any earlier transaction
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _signed_log(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    amount = df["amount"].astype(float)
    old_o = df["old_balance_orig"].astype(float)
    new_o = df["new_balance_orig"].astype(float)
    old_d = df["old_balance_dest"].astype(float)
    new_d = df["new_balance_dest"].astype(float)
    typ = df["type"]

    out = pd.DataFrame(index=df.index)
    out["type"] = typ
    out["amount"] = amount  # kept for rules and reasons, not a model input
    out["amount_log"] = np.log1p(amount)
    out["hour_of_day"] = (df["step"].astype(int) % 24).astype(float)

    positive_bal = old_o > 0
    frac = np.where(positive_bal, amount / old_o.where(positive_bal, 1.0), 0.0)
    out["balance_frac"] = np.clip(frac, 0.0, 2.0)
    is_outflow = typ.isin(OUTFLOW_TYPES)
    out["drains_account"] = (is_outflow & positive_bal & (out["balance_frac"] >= DRAIN_FRACTION)).astype(float)
    out["orig_balance_zero"] = (~positive_bal).astype(float)

    sign = np.where(typ == "CASH_IN", 1.0, -1.0)
    out["err_orig_signed_log"] = _signed_log(pd.Series(old_o + sign * amount - new_o, index=df.index))
    out["err_dest_signed_log"] = _signed_log(pd.Series(old_d + amount - new_d, index=df.index))
    out["dest_old_zero"] = (old_d <= 0).astype(float)

    out["first_time_beneficiary"] = (df["pair_prev_cnt"].astype(float) == 0).astype(float)
    out["vel_cnt"] = df["vel_cnt"].astype(float)
    out["vel_amt_log"] = np.log1p(df["vel_amt"].astype(float))
    prev_cnt = df["sender_prev_cnt"].astype(float)
    prev_avg = df["sender_prev_avg"].astype(float)
    has_hist = (prev_cnt > 0) & prev_avg.notna() & (prev_avg > 0)
    ratio = np.where(has_hist, amount / prev_avg.where(has_hist, 1.0), 1.0)
    out["amount_vs_avg_log"] = np.log(np.clip(ratio, 1e-3, 1e3))
    out["has_history"] = has_hist.astype(float)
    return out
