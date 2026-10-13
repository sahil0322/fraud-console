"""Synthetic stand-in for PaySim, with PaySim's exact column names.

This exists only so the pipeline can be built and tested without the Kaggle file.
It is NOT PaySim and results from it must never be quoted as PaySim results.
Real data goes through the same loader: see fraud/data.py.

Built-in difficulty, so scores are not trivially perfect:
  * some legitimate customers empty their own account
  * some legitimate recipients are not credited in the data (balance mismatch)
  * a share of fraud is "stealthy": partial amounts, known recipient, consistent balances
  * the stealthy share grows over time, so the later test window is harder than training
  * a share of fraud is a short burst of transfers from one victim account
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, RANDOM_SEED

TYPES = np.array(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"])
TYPE_P = np.array([0.34, 0.22, 0.35, 0.08, 0.01])
AMOUNT_LOGN = {  # (mean, sigma) of log(amount) per type
    "PAYMENT": (7.5, 1.1),
    "CASH_IN": (11.5, 0.9),
    "CASH_OUT": (11.0, 1.0),
    "TRANSFER": (11.3, 1.0),
    "DEBIT": (6.5, 1.0),
}
HOUR_W = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 5, 4, 3, 2], dtype=float)
NIGHT_W = np.array([6, 6, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5], dtype=float)


def _steps(rng: np.random.Generator, n: int, weights: np.ndarray, n_steps: int) -> np.ndarray:
    hour = rng.choice(24, size=n, p=weights / weights.sum())
    day = rng.integers(0, (n_steps + 23) // 24, size=n)
    return np.clip(day * 24 + hour + 1, 1, n_steps)


def _legit(rng: np.random.Generator, n: int, n_steps: int, n_cust: int, n_merch: int) -> pd.DataFrame:
    sender = rng.choice(n_cust, size=n, p=_activity(rng, n_cust))
    base_bal = rng.lognormal(10.0, 1.5, n_cust)
    typ = rng.choice(TYPES, size=n, p=TYPE_P)
    step = _steps(rng, n, HOUR_W, n_steps)

    amount = np.empty(n)
    for t, (mu, sg) in AMOUNT_LOGN.items():
        m = typ == t
        amount[m] = rng.lognormal(mu, sg, m.sum())
    old = base_bal[sender] * rng.uniform(0.4, 1.6, n)

    is_out = np.isin(typ, ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"])
    cap = old * rng.uniform(0.05, 0.98, n)
    amount = np.where(is_out, np.minimum(amount, cap), amount)
    # about 4% of legitimate cash-outs and transfers empty the whole account
    drain = np.isin(typ, ["CASH_OUT", "TRANSFER"]) & (rng.random(n) < 0.04)
    amount = np.where(drain, old, amount)

    new = np.where(typ == "CASH_IN", old + amount, old - amount)
    stale = rng.random(n) < 0.01  # sender balance not updated in the record
    new = np.where(stale, old, new)

    base_dest = rng.lognormal(10.5, 1.6, n_cust)
    fav_k = rng.integers(0, 3, n)
    fav = (sender * 7919 + fav_k * 104729 + 13) % n_cust
    rand = rng.integers(0, n_cust, n)
    dest_c = np.where(rng.random(n) < 0.55, fav, rand)
    is_merchant = np.isin(typ, ["PAYMENT", "DEBIT"])
    dest_m = rng.integers(0, n_merch, n)
    name_dest = np.where(is_merchant, ["M%d" % i for i in dest_m], ["C%d" % i for i in dest_c])

    old_d = base_dest[dest_c] * rng.uniform(0.5, 1.5, n)
    old_d = np.where(rng.random(n) < 0.25, 0.0, old_d)
    new_d = old_d + amount
    not_credited = rng.random(n) < 0.08
    new_d = np.where(not_credited, old_d, new_d)
    old_d = np.where(is_merchant, 0.0, old_d)
    new_d = np.where(is_merchant, 0.0, new_d)

    return pd.DataFrame({
        "step": step, "type": typ, "amount": amount.round(2),
        "nameOrig": ["C%d" % i for i in sender],
        "oldbalanceOrg": old.round(2), "newbalanceOrig": new.round(2),
        "nameDest": name_dest,
        "oldbalanceDest": old_d.round(2), "newbalanceDest": new_d.round(2),
        "isFraud": 0,
    })


def _activity(rng: np.random.Generator, n_cust: int) -> np.ndarray:
    w = rng.lognormal(0.0, 1.0, n_cust)
    return w / w.sum()


def _fraud(rng: np.random.Generator, n_target: int, n_steps: int, n_cust: int) -> pd.DataFrame:
    n_events = max(1, int(n_target / 1.5))  # mean rows per event is 1.5 with 25% bursts of 3
    ev_step = _steps(rng, n_events, NIGHT_W, n_steps)
    victims = rng.integers(0, n_cust, n_events)
    rows = []
    fresh = 0
    for e in range(n_events):
        t = ev_step[e] / n_steps
        stealth = rng.random() < (0.10 + 0.25 * t)
        burst = rng.random() < 0.25
        bal = float(rng.lognormal(11.5, 1.0))
        for r in range(3 if burst else 1):
            step = int(min(ev_step[e] + r, n_steps))
            if stealth:
                frac = rng.uniform(0.15, 0.65)
            elif burst and r < 2:
                frac = rng.uniform(0.3, 0.5)
            elif burst:
                frac = rng.uniform(0.9, 1.0)
            else:
                frac = rng.uniform(0.97, 1.0) if rng.random() < 0.85 else rng.uniform(0.5, 0.95)
            amount = round(bal * frac, 2)
            typ = "TRANSFER" if rng.random() < 0.5 else "CASH_OUT"
            old = round(bal, 2)
            new = round(bal - amount, 2) if rng.random() < 0.97 else old
            bal = max(bal - amount, 0.0)
            if stealth:
                dest = "C%d" % rng.integers(0, n_cust)
                old_d = round(float(rng.lognormal(10.5, 1.6)), 2)
                new_d = round(old_d + amount, 2)
            else:
                fresh += 1
                dest = "C9%07d" % fresh
                old_d = 0.0
                new_d = 0.0 if rng.random() < 0.7 else amount
            rows.append((step, typ, amount, "C%d" % victims[e], old, new, dest, old_d, new_d, 1))
    cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
    return pd.DataFrame(rows, columns=cols)


def generate(n_rows: int = 400_000, fraud_rate: float = 0.0025, seed: int = RANDOM_SEED,
             n_steps: int = 743) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_cust = max(1000, n_rows // 6)
    legit = _legit(rng, n_rows - n_fraud, n_steps, n_cust, n_merch=max(300, n_cust // 20))
    fraud = _fraud(rng, n_fraud, n_steps, n_cust)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    # PaySim's own flag fires on very few rows and is derived from the label: mimic that so
    # the loader's "drop isFlaggedFraud" step has something real to drop.
    flag = (df["isFraud"] == 1) & (df["type"] == "TRANSFER") & (df["amount"] > 200_000)
    df["isFlaggedFraud"] = (flag & (rng.random(len(df)) < 0.05)).astype(int)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=400_000)
    p.add_argument("--fraud-rate", type=float, default=0.0025)
    p.add_argument("--out", type=Path, default=DATA_DIR / "synthetic_paysim.csv")
    a = p.parse_args()
    df = generate(a.rows, a.fraud_rate)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows to {a.out} (fraud rate {df['isFraud'].mean():.4%})")


if __name__ == "__main__":
    main()
