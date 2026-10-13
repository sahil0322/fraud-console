"""Evaluation for rare-event detection. Accuracy is not used anywhere in this project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import RANDOM_SEED, RULE_RISK


def _order(score: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """Indices from highest to lowest score. Ties are broken at random with a fixed seed,
    so a rules-only baseline with many equal scores is not helped or hurt by row order."""
    rng = np.random.default_rng(seed)
    return np.lexsort((rng.random(len(score)), -np.asarray(score, dtype=float)))


def budget_metrics(y: np.ndarray, score: np.ndarray, frac: float) -> dict:
    y = np.asarray(y)
    k = max(1, int(round(frac * len(y))))
    top = _order(score)[:k]
    caught = int(y[top].sum())
    total = int(y.sum())
    return {
        "budget_frac": frac,
        "budget_n": k,
        "recall": caught / total if total else float("nan"),
        "precision": caught / k,
    }


def summarize(y: np.ndarray, score: np.ndarray, frac: float) -> dict:
    y = np.asarray(y)
    b = budget_metrics(y, score, frac)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "recall_at_budget": b["recall"],
        "precision_at_budget": b["precision"],
    }


def candidate_thresholds() -> np.ndarray:
    fine = [0.995, 0.999, 0.9995]  # class-weighted models push scores close to 1
    return np.unique(np.concatenate([np.linspace(0.02, 0.99, 98), fine, list(RULE_RISK.values())]))


def cost_table(y: np.ndarray, risk: np.ndarray, c_missed: float, c_false: float,
               thresholds: np.ndarray | None = None) -> pd.DataFrame:
    y = np.asarray(y).astype(bool)
    risk = np.asarray(risk, dtype=float)
    rows = []
    total_fraud = int(y.sum())
    for t in (candidate_thresholds() if thresholds is None else thresholds):
        alert = risk >= t
        tp = int((alert & y).sum())
        fp = int((alert & ~y).sum())
        fn = total_fraud - tp
        n_alert = tp + fp
        rows.append({
            "threshold": float(t), "alerts": n_alert, "alert_rate": n_alert / len(y),
            "true_alerts": tp, "false_alerts": fp, "missed_fraud": fn,
            "precision": tp / n_alert if n_alert else 0.0,
            "recall": tp / total_fraud if total_fraud else 0.0,
            "cost": fn * c_missed + fp * c_false,
        })
    return pd.DataFrame(rows)


def pick_review_threshold(table: pd.DataFrame) -> float:
    """Lowest total cost; on a tie, the threshold that raises fewer alerts."""
    best = table.sort_values(["cost", "alerts", "threshold"], ascending=[True, True, False]).iloc[0]
    return float(best["threshold"])
