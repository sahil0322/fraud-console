"""Load the model's test-window alerts into the database for the console.

    python -m scripts.load_alerts --csv data/synthetic_paysim.csv
    DATABASE_URL=postgresql://user:pass@localhost:5432/fraud python -m scripts.load_alerts --csv ...

Run `python -m fraud.train --csv ...` first. The CSV must be the same file the model was
trained on; this is checked against a fingerprint saved by training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from fraud import config as C
from fraud.data import load_paysim
from fraud.db import connect
from fraud.service import utcnow

COLS = ["step", "type", "amount", "name_orig", "old_balance_orig", "new_balance_orig",
        "name_dest", "old_balance_dest", "new_balance_dest"]

INSERT = (
    "INSERT INTO alerts (id, step, type, amount, name_orig, old_balance_orig, new_balance_orig, "
    "name_dest, old_balance_dest, new_balance_dest, risk, model_risk, rule_risk, reasons, is_fraud, "
    "status, created_at) VALUES (:id, :step, :type, :amount, :name_orig, :old_balance_orig, "
    ":new_balance_orig, :name_dest, :old_balance_dest, :new_balance_dest, :risk, :model_risk, "
    ":rule_risk, :reasons, :is_fraud, 'open', :now)"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--reset", action="store_true", help="delete existing alerts and actions first")
    a = p.parse_args()

    for f in (C.METRICS_PATH, C.TEST_SCORED_PATH):
        if not f.exists():
            sys.exit(f"{f} not found. Run: python -m fraud.train --csv {a.csv}")
    metrics = json.loads(C.METRICS_PATH.read_text())
    df = load_paysim(a.csv)
    if metrics["fingerprint"] != f"{len(df)}:{df['amount'].sum():.2f}":
        sys.exit("This CSV is not the file the model was trained on. Retrain on it first.")

    scored = pd.read_csv(C.TEST_SCORED_PATH)
    rows = scored.merge(df, on="txn_id", how="left", validate="one_to_one")
    if rows[COLS].isna().any().any():
        sys.exit("Some alerts did not match a transaction in the CSV.")

    url = os.environ.get("DATABASE_URL", C.DEFAULT_DB_URL)
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    db = connect(url)
    db.init_schema()
    if a.reset:
        db.query("DELETE FROM analyst_actions")
        db.query("DELETE FROM alerts")
    existing = db.query("SELECT COUNT(*) AS n FROM alerts")[0]["n"]
    if existing:
        sys.exit(f"{existing} alerts already loaded. Use --reset to replace them.")

    now = utcnow()
    with db.transaction() as tx:
        for r in rows.to_dict("records"):
            tx.query(INSERT, {
                "id": int(r["txn_id"]), "step": int(r["step"]), "type": r["type"], "amount": float(r["amount"]),
                "name_orig": r["name_orig"], "old_balance_orig": float(r["old_balance_orig"]),
                "new_balance_orig": float(r["new_balance_orig"]), "name_dest": r["name_dest"],
                "old_balance_dest": float(r["old_balance_dest"]), "new_balance_dest": float(r["new_balance_dest"]),
                "risk": float(r["risk"]), "model_risk": float(r["model_risk"]), "rule_risk": float(r["rule_risk"]),
                "reasons": r["reasons"], "is_fraud": int(r["is_fraud"]), "now": now,
            })
    print(f"loaded {len(rows):,} alerts into {url.split('@')[-1]}")


if __name__ == "__main__":
    main()
