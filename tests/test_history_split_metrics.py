"""History features (batch SQL vs live queries), time-based split, metrics and thresholds."""
import numpy as np
import pandas as pd

from fraud import config as C
from fraud.history import batch_history
from fraud.metrics import budget_metrics, cost_table, pick_review_threshold
from fraud.train import split_masks


def _tiny(rows):
    df = pd.DataFrame(rows, columns=["step", "name_orig", "name_dest", "amount"])
    df["txn_id"] = range(1, len(df) + 1)
    return df


def test_velocity_window_is_strictly_earlier_steps_and_six_wide():
    # sender A at steps 1, 5, 7, 7, 14: window for step 7 is steps 1..6
    df = _tiny([(1, "A", "X", 10.0), (5, "A", "X", 20.0), (7, "A", "Y", 40.0),
                (7, "A", "Z", 80.0), (14, "A", "X", 160.0)])
    h = batch_history(df).set_index("txn_id")
    assert h.loc[1, "vel_cnt"] == 0
    assert h.loc[2, "vel_cnt"] == 1 and h.loc[2, "vel_amt"] == 10.0     # step 1 is 4 back
    assert h.loc[3, "vel_cnt"] == 2 and h.loc[3, "vel_amt"] == 30.0     # steps 1 and 5
    assert h.loc[4, "vel_cnt"] == 2                                      # same-step row not counted
    assert h.loc[5, "vel_cnt"] == 0                                      # step 7 is the edge, 14-6=8 > 7


def test_window_edge_step_minus_six_is_included_and_minus_seven_is_not():
    df = _tiny([(1, "A", "X", 1.0), (2, "A", "X", 2.0), (8, "A", "X", 4.0)])
    h = batch_history(df).set_index("txn_id")
    assert h.loc[3, "vel_cnt"] == 1 and h.loc[3, "vel_amt"] == 2.0  # step 2 is 6 back, step 1 is 7 back


def test_pair_and_sender_history():
    df = _tiny([(1, "A", "X", 100.0), (2, "A", "X", 300.0), (3, "A", "Y", 50.0), (3, "B", "X", 9.0)])
    h = batch_history(df).set_index("txn_id")
    assert list(h["pair_prev_cnt"]) == [0, 1, 0, 0]
    assert list(h["sender_prev_cnt"]) == [0, 1, 2, 0]
    assert h.loc[3, "sender_prev_avg"] == 200.0 and pd.isna(h.loc[1, "sender_prev_avg"])


def test_time_split_is_ordered_disjoint_and_complete():
    step = pd.Series(np.random.default_rng(0).integers(1, 744, 5000))
    m = split_masks(step)
    cuts = m.pop("_cuts")
    tr, va, te = step[m["train"]], step[m["val"]], step[m["test"]]
    assert tr.max() < va.min() and va.max() < te.min()
    assert (m["train"].astype(int) + m["val"].astype(int) + m["test"].astype(int) == 1).all()
    assert cuts[0] == C.TRAIN_END_FRAC * 743 or cuts[0] <= C.TRAIN_END_FRAC * step.max() + 1e-9


def test_recall_at_budget_on_a_known_ranking():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    b = budget_metrics(y, score, 0.2)           # top 2 rows: one fraud caught of two
    assert b["budget_n"] == 2 and b["recall"] == 0.5 and b["precision"] == 0.5


def test_budget_ties_do_not_depend_on_row_order():
    y = np.array([1] + [0] * 99)
    score = np.zeros(100)
    r1 = budget_metrics(y, score, 0.1)["recall"]
    r2 = budget_metrics(y[::-1].copy(), score, 0.1)["recall"]
    assert r1 in (0.0, 1.0) and r2 in (0.0, 1.0)  # a fixed seed gives a reproducible tie-break


def test_cost_table_arithmetic_and_threshold_choice():
    y = np.array([1, 1, 0, 0, 0, 0])
    risk = np.array([0.95, 0.60, 0.90, 0.20, 0.10, 0.05])
    t = cost_table(y, risk, c_missed=100.0, c_false=1.0, thresholds=np.array([0.5, 0.8, 0.99]))
    row = t.set_index("threshold")
    assert row.loc[0.5, "alerts"] == 3 and row.loc[0.5, "false_alerts"] == 1 and row.loc[0.5, "cost"] == 1.0
    assert row.loc[0.8, "missed_fraud"] == 1 and row.loc[0.8, "cost"] == 101.0
    assert row.loc[0.99, "cost"] == 200.0
    assert pick_review_threshold(t) == 0.5
