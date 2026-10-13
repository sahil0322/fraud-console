"""Risk score, review flag and top-three reasons from the logistic regression and the rules.

Risk is a ranking score between 0 and 1. The model is trained with class weights, so the
score is not a calibrated probability: 0.9 does not mean "90% chance of fraud".

Reasons: fired rules first, in plain words, then the largest positive contributions of the
logistic regression (coefficient times standardised feature value).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, VELOCITY_WINDOW_STEPS
from .features import FEATURES
from .rules import RULES, rule_hits, rule_risk

W = VELOCITY_WINDOW_STEPS


def _yes_no(v: float) -> str:
    return "yes" if v >= 0.5 else "no"


def _unlog(v: float) -> float:
    return float(np.sign(v) * np.expm1(abs(v)))


# feature -> (label, value formatter taking the engineered row, is_continuous)
FEATURE_TEXT = {
    "amount_log": ("Transaction amount", lambda r: f"{r['amount']:,.2f}", True),
    "hour_of_day": ("Hour of day", lambda r: f"{int(r['hour_of_day']):02d}:00", True),
    "balance_frac": ("Amount as a share of the sender's balance", lambda r: f"{r['balance_frac']:.0%}", True),
    "drains_account": ("Outflow empties the sender's account", lambda r: _yes_no(r["drains_account"]), False),
    "orig_balance_zero": ("Sender balance was zero before", lambda r: _yes_no(r["orig_balance_zero"]), False),
    "err_orig_signed_log": ("Gap between expected and recorded sender balance",
                            lambda r: f"{_unlog(r['err_orig_signed_log']):,.2f}", True),
    "err_dest_signed_log": ("Gap between expected and recorded recipient balance",
                            lambda r: f"{_unlog(r['err_dest_signed_log']):,.2f}", True),
    "dest_old_zero": ("Recipient balance was zero before", lambda r: _yes_no(r["dest_old_zero"]), False),
    "first_time_beneficiary": ("First payment from this sender to this recipient",
                               lambda r: _yes_no(r["first_time_beneficiary"]), False),
    "vel_cnt": (f"Sender transactions in the previous {W} hours", lambda r: f"{int(r['vel_cnt'])}", True),
    "vel_amt_log": (f"Amount sent in the previous {W} hours", lambda r: f"{np.expm1(r['vel_amt_log']):,.2f}", True),
    "amount_vs_avg_log": ("Amount relative to the sender's own average",
                          lambda r: f"{np.exp(r['amount_vs_avg_log']):.1f} times", True),
    "has_history": ("Sender has earlier transactions", lambda r: _yes_no(r["has_history"]), False),
}


class Scorer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipe = bundle["lr"]
        self.threshold = float(bundle["threshold"])
        self.version = bundle["version"]
        self._names = list(self.pipe.named_steps["pre"].get_feature_names_out())
        self._coef = self.pipe.named_steps["clf"].coef_[0]

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "Scorer":
        return cls(joblib.load(path))

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        m = self.pipe.predict_proba(feats[FEATURES])[:, 1]
        r = rule_risk(feats).to_numpy()
        risk = np.maximum(m, r)
        return pd.DataFrame({"model_risk": m, "rule_risk": r, "risk": risk,
                             "flagged": risk >= self.threshold}, index=feats.index)

    def explain(self, feats: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Up to `top` reasons per row: fired rules first, then the biggest model drivers."""
        if len(feats) == 0:
            return []
        xt = self.pipe.named_steps["pre"].transform(feats[FEATURES])
        xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
        contrib = xt * self._coef
        hits = rule_hits(feats)
        ordered_rules = sorted(RULES, key=lambda r: -r.risk)
        out = []
        for i in range(len(feats)):
            reasons = [{"source": "rule", "code": r.code, "text": r.text}
                       for r in ordered_rules if hits.iloc[i][r.code]]
            row = feats.iloc[i]
            for j in np.argsort(-contrib[i]):
                if len(reasons) >= top or contrib[i, j] <= 0:
                    break
                reasons.append(self._model_reason(self._names[j], row, bool(xt[i, j] > 0)))
            out.append(reasons[:top])
        return out

    @staticmethod
    def _model_reason(name: str, row: pd.Series, above: bool) -> dict:
        if name.startswith("cat__type_"):
            return {"source": "model", "code": "type", "text": f"Transaction type is {row['type']}"}
        feat = name[len("num__"):]
        label, fmt, continuous = FEATURE_TEXT[feat]
        text = f"{label}: {fmt(row)}"
        if continuous:
            text += " (above the training average)" if above else " (below the training average)"
        return {"source": "model", "code": feat, "text": text}
