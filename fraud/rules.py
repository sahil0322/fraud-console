"""Readable rules that run next to the model.

Each rule is a plain boolean condition on the engineered feature frame. The final risk
of a transaction is the higher of the rule risk and the model risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import DRAIN_FRACTION, LARGE_AMOUNT, OUTFLOW_TYPES, RULE_RISK, VELOCITY_COUNT, VELOCITY_WINDOW_STEPS


@dataclass(frozen=True)
class Rule:
    code: str
    risk: float
    text: str
    test: Callable[[pd.DataFrame], pd.Series]


RULES: list[Rule] = [
    Rule(
        "DRAIN_ACCOUNT", RULE_RISK["DRAIN_ACCOUNT"],
        f"Outflow takes at least {DRAIN_FRACTION:.0%} of the sender's balance",
        lambda f: f["drains_account"] == 1,
    ),
    Rule(
        "NEW_RECIPIENT_LARGE", RULE_RISK["NEW_RECIPIENT_LARGE"],
        f"Large outflow (at least {LARGE_AMOUNT:,.0f}) to an account with no prior balance",
        lambda f: f["type"].isin(OUTFLOW_TYPES) & (f["dest_old_zero"] == 1) & (f["amount"] >= LARGE_AMOUNT),
    ),
    Rule(
        "HIGH_VELOCITY", RULE_RISK["HIGH_VELOCITY"],
        f"{VELOCITY_COUNT} or more transactions from this sender in the previous {VELOCITY_WINDOW_STEPS} hours",
        lambda f: f["vel_cnt"] >= VELOCITY_COUNT,
    ),
]


def rule_hits(feats: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame, one column per rule code."""
    return pd.DataFrame({r.code: r.test(feats).fillna(False).astype(bool) for r in RULES}, index=feats.index)


def rule_risk(feats: pd.DataFrame) -> pd.Series:
    """Highest risk among the rules that fired, 0.0 if none fired."""
    hits = rule_hits(feats)
    risks = np.array([r.risk for r in RULES])
    scored = hits.to_numpy() * risks  # False -> 0
    return pd.Series(scored.max(axis=1) if len(RULES) else 0.0, index=feats.index)


def fired(feats_row: pd.DataFrame) -> list[Rule]:
    """Rules that fired for a one-row frame, highest risk first."""
    hits = rule_hits(feats_row).iloc[0]
    return sorted((r for r in RULES if hits[r.code]), key=lambda r: -r.risk)
