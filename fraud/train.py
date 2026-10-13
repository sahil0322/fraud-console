"""Train logistic regression, compare it with a rules-only baseline, tune the threshold by cost.

    python -m fraud.train --csv data/synthetic_paysim.csv
    python -m fraud.train --csv path/to/PaySim.csv

Windows by step (hours), never random:
    train       earliest 60% of the step range      fit the model
    validation  next 15%                            choose the threshold only
    test        last 25%                            reported once, no choice made on it
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .data import load_paysim
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, engineer
from .history import attach_history
from .metrics import cost_table, pick_review_threshold, summarize
from .rules import rule_risk
from .scoring import Scorer


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


def split_masks(step: pd.Series) -> dict:
    top = float(step.max())
    train_end, val_end = C.TRAIN_END_FRAC * top, C.VAL_END_FRAC * top
    return {"train": step <= train_end, "val": (step > train_end) & (step <= val_end),
            "test": step > val_end, "_cuts": (train_end, val_end, top)}


def _py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def run(csv: Path, nrows: int | None, c_missed: float, c_false: float) -> dict:
    t0 = time.time()
    source = "synthetic" if "synthetic" in csv.name.lower() else "paysim"
    print(f"[1/5] loading {csv} (source label: {source})")
    df = load_paysim(csv, nrows=nrows)
    print(f"      {len(df):,} rows, fraud rate {df['is_fraud'].mean():.4%}, "
          f"distinct senders / rows = {df['name_orig'].nunique() / len(df):.3f}")

    print("[2/5] history features with SQL window functions")
    df = attach_history(df)
    feats = engineer(df)
    y = df["is_fraud"].to_numpy()

    m = split_masks(df["step"])
    cuts = m.pop("_cuts")
    sizes = {k: {"rows": int(v.sum()), "fraud": int(y[v.to_numpy()].sum())} for k, v in m.items()}
    for k, s in sizes.items():
        print(f"      {k:5s} rows {s['rows']:>9,}  fraud {s['fraud']:>6,}  rate {s['fraud'] / max(s['rows'], 1):.4%}")
        if s["fraud"] < 20:
            sys.exit(f"Only {s['fraud']} fraud rows in the {k} window; need at least 20. Use more data.")

    print("[3/5] fitting logistic regression on the train window")
    pipe = make_pipeline().fit(feats.loc[m["train"], FEATURES], y[m["train"].to_numpy()])

    def risks(name: str) -> dict:
        f = feats.loc[m[name]]
        r = rule_risk(f).to_numpy()
        p = pipe.predict_proba(f[FEATURES])[:, 1]
        return {"rules": r, "lr": p, "lr+rules": np.maximum(p, r)}

    val_r, test_r = risks("val"), risks("test")
    yv, yt = y[m["val"].to_numpy()], y[m["test"].to_numpy()]

    print("[4/5] choosing the review threshold on the validation window")
    table = cost_table(yv, val_r["lr+rules"], c_missed, c_false)
    thr = pick_review_threshold(table)
    print(f"      threshold {thr:.3f} (cost of a missed fraud {c_missed:g}, of a false alert {c_false:g})")

    bundle = {"version": f"lr-{source}-{dt.datetime.now(dt.timezone.utc):%Y%m%d}", "lr": pipe,
              "threshold": thr, "source": source, "sklearn": sklearn.__version__}
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, C.MODEL_PATH)
    scorer = Scorer(bundle)

    print("[5/5] evaluating on the test window and writing artifacts")
    comparison = {k: summarize(yt, test_r[k], C.REVIEW_BUDGET_FRAC) for k in ("rules", "lr", "lr+rules")}
    ts = scorer.score(feats.loc[m["test"]])
    flagged = ts["flagged"].to_numpy()
    tp, n_flag = int((flagged & (yt == 1)).sum()), int(flagged.sum())
    n_fraud = int(yt.sum())
    at_thr = {
        "alerts": n_flag, "flagged_share": n_flag / len(yt),
        "alert_precision": tp / n_flag if n_flag else 0.0, "fraud_recall": tp / n_fraud,
        "cost": float((n_fraud - tp) * c_missed + (n_flag - tp) * c_false),
        "cost_if_nothing_flagged": float(n_fraud * c_missed),
        "cost_if_everything_flagged": float((len(yt) - n_fraud) * c_false),
    }
    # rules-only at its own natural operating point: every row where any rule fires
    rflag = test_r["rules"] > 0
    rtp = int((rflag & (yt == 1)).sum())
    rules_at = {"flagged_share": float(rflag.mean()),
                "alert_precision": rtp / int(rflag.sum()) if rflag.any() else 0.0,
                "fraud_recall": rtp / n_fraud}

    idx = feats.loc[m["test"]].index[flagged]
    reasons = scorer.explain(feats.loc[idx])
    pd.DataFrame({
        "txn_id": df.loc[idx, "txn_id"].to_numpy(),
        "model_risk": ts.loc[idx, "model_risk"].to_numpy(),
        "rule_risk": ts.loc[idx, "rule_risk"].to_numpy(),
        "risk": ts.loc[idx, "risk"].to_numpy(),
        "reasons": [json.dumps(r) for r in reasons],
    }).to_csv(C.TEST_SCORED_PATH, index=False)
    table.to_csv(C.THRESHOLD_TABLE_PATH, index=False)

    metrics = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": source, "csv": csv.name, "model_version": bundle["version"],
        "fingerprint": f"{len(df)}:{df['amount'].sum():.2f}",
        "rows": len(df), "fraud_rate": float(df["is_fraud"].mean()),
        "distinct_sender_ratio": float(df["name_orig"].nunique() / len(df)),
        "n_features": len(FEATURES), "features": FEATURES,
        "windows": sizes, "step_cuts": {"train_end": cuts[0], "val_end": cuts[1], "max_step": cuts[2]},
        "review_budget_frac": C.REVIEW_BUDGET_FRAC, "test_comparison": comparison,
        "threshold": thr, "test_at_threshold": at_thr, "test_rules_only_at_natural_point": rules_at,
        "cost_assumptions": {"missed_fraud": c_missed, "false_alert": c_false},
        "sklearn": sklearn.__version__, "train_seconds": round(time.time() - t0, 1),
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=_py))
    print_report(metrics)
    return metrics


def print_report(m: dict) -> None:
    b = int(round(m["review_budget_frac"] * 100))
    print("\nTest window (later in time than anything the model saw)")
    print(f"{'':10s} {'PR-AUC':>8s} {'recall@' + str(b) + '%':>10s} {'precision@' + str(b) + '%':>13s}")
    for k, v in m["test_comparison"].items():
        print(f"{k:10s} {v['pr_auc']:8.3f} {v['recall_at_budget']:10.3f} {v['precision_at_budget']:13.3f}")
    a, r = m["test_at_threshold"], m["test_rules_only_at_natural_point"]
    print(f"\nAt the chosen threshold ({m['threshold']:.3f}): review {a['flagged_share']:.2%} of transactions, "
          f"catch {a['fraud_recall']:.1%} of fraud, alert precision {a['alert_precision']:.1%}")
    print(f"Rules only (any rule fires): review {r['flagged_share']:.2%}, catch {r['fraud_recall']:.1%}, "
          f"precision {r['alert_precision']:.1%}")
    print(f"Cost {a['cost']:,.0f} vs {a['cost_if_nothing_flagged']:,.0f} (flag nothing) "
          f"vs {a['cost_if_everything_flagged']:,.0f} (flag everything)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--nrows", type=int, default=None, help="read only the first N rows (quick tests)")
    p.add_argument("--cost-missed", type=float, default=C.COST_MISSED_FRAUD)
    p.add_argument("--cost-false", type=float, default=C.COST_FALSE_ALERT)
    a = p.parse_args()
    if not a.csv.exists():
        sys.exit(f"{a.csv} not found. For the synthetic stand-in run: python -m fraud.synth")
    run(a.csv, a.nrows, a.cost_missed, a.cost_false)


if __name__ == "__main__":
    main()
