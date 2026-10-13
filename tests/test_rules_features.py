"""Rule engine and feature engineering."""
import numpy as np

from fraud.config import RULE_RISK
from fraud.data import load_paysim
from fraud.features import FEATURES, NUMERIC_FEATURES, engineer
from fraud.rules import fired, rule_hits, rule_risk

from .helpers import raw_frame, raises


def feats(**over):
    return engineer(raw_frame([over]))


def test_drain_rule_fires_on_outflow_that_empties_account():
    f = feats(type="TRANSFER", amount=99000.0, old_balance_orig=100000.0, new_balance_orig=1000.0)
    assert rule_hits(f).loc[0, "DRAIN_ACCOUNT"]
    assert rule_risk(f).iloc[0] == RULE_RISK["DRAIN_ACCOUNT"]


def test_drain_rule_ignores_payments_cash_in_and_zero_balance():
    for kw in (dict(type="PAYMENT", amount=100000.0, new_balance_orig=0.0),
               dict(type="CASH_IN", amount=100000.0, new_balance_orig=200000.0),
               dict(type="TRANSFER", amount=500.0, old_balance_orig=0.0, new_balance_orig=0.0)):
        assert not rule_hits(feats(**kw)).loc[0, "DRAIN_ACCOUNT"], kw


def test_drain_threshold_is_inclusive_at_95_percent():
    assert rule_hits(feats(amount=95000.0, old_balance_orig=100000.0)).loc[0, "DRAIN_ACCOUNT"]
    assert not rule_hits(feats(amount=94000.0, old_balance_orig=100000.0)).loc[0, "DRAIN_ACCOUNT"]


def test_new_recipient_large_rule():
    big = feats(type="CASH_OUT", amount=250000.0, old_balance_orig=900000.0, new_balance_orig=650000.0,
                old_balance_dest=0.0, new_balance_dest=0.0)
    assert rule_hits(big).loc[0, "NEW_RECIPIENT_LARGE"]
    small = feats(type="CASH_OUT", amount=1000.0, old_balance_dest=0.0, new_balance_dest=0.0)
    assert not rule_hits(small).loc[0, "NEW_RECIPIENT_LARGE"]
    known = feats(type="CASH_OUT", amount=250000.0, old_balance_orig=900000.0, old_balance_dest=5000.0)
    assert not rule_hits(known).loc[0, "NEW_RECIPIENT_LARGE"]


def test_velocity_rule():
    assert rule_hits(feats(vel_cnt=3)).loc[0, "HIGH_VELOCITY"]
    assert not rule_hits(feats(vel_cnt=2)).loc[0, "HIGH_VELOCITY"]


def test_rule_risk_is_the_highest_fired_and_zero_when_none():
    f = feats(type="TRANSFER", amount=99000.0, old_balance_orig=100000.0, vel_cnt=5)
    assert rule_risk(f).iloc[0] == max(RULE_RISK["DRAIN_ACCOUNT"], RULE_RISK["HIGH_VELOCITY"])
    assert [r.code for r in fired(f)][0] == "DRAIN_ACCOUNT"
    assert rule_risk(feats()).iloc[0] == 0.0


def test_features_have_exact_columns_and_no_missing_values():
    f = engineer(raw_frame([{}, {"sender_prev_avg": 2000.0, "sender_prev_cnt": 4, "pair_prev_cnt": 2},
                            {"old_balance_orig": 0.0, "new_balance_orig": 0.0, "type": "PAYMENT"}]))
    assert set(FEATURES) <= set(f.columns)
    assert not f[NUMERIC_FEATURES].isna().any().any()
    assert np.isfinite(f[NUMERIC_FEATURES].to_numpy()).all()


def test_features_never_include_label_columns():
    f = feats()
    for bad in ("is_fraud", "isFraud", "isFlaggedFraud", "is_flagged_fraud"):
        assert bad not in f.columns and bad not in FEATURES


def test_balance_error_is_zero_for_a_consistent_transaction():
    f = feats(type="TRANSFER", amount=5000.0, old_balance_orig=100000.0, new_balance_orig=95000.0,
              old_balance_dest=20000.0, new_balance_dest=25000.0)
    assert f.loc[0, "err_orig_signed_log"] == 0.0 and f.loc[0, "err_dest_signed_log"] == 0.0


def test_balance_error_detects_uncredited_recipient():
    f = feats(amount=5000.0, old_balance_dest=20000.0, new_balance_dest=20000.0)
    assert f.loc[0, "err_dest_signed_log"] > 0


def test_cash_in_balance_check_uses_opposite_sign():
    f = feats(type="CASH_IN", amount=5000.0, old_balance_orig=10000.0, new_balance_orig=15000.0)
    assert f.loc[0, "err_orig_signed_log"] == 0.0


def test_first_time_beneficiary_and_history_flags():
    assert feats(pair_prev_cnt=0).loc[0, "first_time_beneficiary"] == 1
    assert feats(pair_prev_cnt=3).loc[0, "first_time_beneficiary"] == 0
    assert feats().loc[0, "has_history"] == 0
    g = feats(sender_prev_cnt=5, sender_prev_avg=2500.0, amount=5000.0)
    assert g.loc[0, "has_history"] == 1
    assert abs(g.loc[0, "amount_vs_avg_log"] - np.log(2.0)) < 1e-9


def test_hour_of_day_comes_from_step():
    assert feats(step=25).loc[0, "hour_of_day"] == 1
    assert feats(step=48).loc[0, "hour_of_day"] == 0


def test_loader_drops_isflaggedfraud_and_checks_columns():
    import tempfile, pathlib, pandas as pd
    from fraud.data import SchemaError
    d = pathlib.Path(tempfile.mkdtemp())
    good = pd.DataFrame({"step": [1, 2], "type": ["TRANSFER", "PAYMENT"], "amount": [1.0, 2.0],
                         "nameOrig": ["C1", "C2"], "oldbalanceOrg": [1.0, 2.0], "newbalanceOrig": [0.0, 0.0],
                         "nameDest": ["C3", "M1"], "oldbalanceDest": [0.0, 0.0], "newbalanceDest": [1.0, 0.0],
                         "isFraud": [1, 0], "isFlaggedFraud": [1, 0]})
    good.to_csv(d / "ok.csv", index=False)
    df = load_paysim(d / "ok.csv")
    assert "isFlaggedFraud" not in df.columns and "is_flagged_fraud" not in df.columns
    assert list(df["txn_id"]) == [1, 2]
    good.drop(columns=["oldbalanceOrg"]).to_csv(d / "bad.csv", index=False)
    e = raises(SchemaError, load_paysim, d / "bad.csv")
    assert "oldbalanceOrg" in str(e)
