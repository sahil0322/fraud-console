"""Shared constants. Everything tunable lives in this one file.

The numbers in the "assumptions" block are choices, not measurements. Change them,
retrain, and re-read the README section on thresholds.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
THRESHOLD_TABLE_PATH = ARTIFACT_DIR / "threshold_costs.csv"
TEST_SCORED_PATH = ARTIFACT_DIR / "test_scored.csv"

# PaySim column names on the left, internal snake_case names on the right.
# isFlaggedFraud is deliberately absent: it is dropped on load because it leaks the label.
PAYSIM_RENAME = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
}
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
OUTFLOW_TYPES = ["CASH_OUT", "TRANSFER"]  # the only types that carry fraud in PaySim

# Time-based split on the `step` column (one step is one hour).
TRAIN_END_FRAC = 0.60  # steps up to 60% of the maximum go to training
VAL_END_FRAC = 0.75    # the next 15% is used only to choose thresholds
# the remaining 25% is the test window and is never used for any choice

# Velocity window, in steps (hours). Only strictly earlier steps are counted, so the
# feature is identical in batch training and in live scoring.
VELOCITY_WINDOW_STEPS = 6

# Rule thresholds and the risk each rule assigns (assumptions, tune on validation data).
DRAIN_FRACTION = 0.95        # amount at or above this share of the balance counts as draining
LARGE_AMOUNT = 200_000.0     # "large" amount in the new-recipient rule
VELOCITY_COUNT = 3           # prior transactions in the window that trigger the velocity rule
RULE_RISK = {"DRAIN_ACCOUNT": 0.70, "NEW_RECIPIENT_LARGE": 0.60, "HIGH_VELOCITY": 0.55}

# Cost assumptions for threshold tuning (units are arbitrary, only the ratio matters).
COST_MISSED_FRAUD = 1000.0
COST_FALSE_ALERT = 10.0

# Review budget used for the "recall at a fixed review budget" metric.
REVIEW_BUDGET_FRAC = 0.01

RANDOM_SEED = 42
DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'alerts.db'}"
