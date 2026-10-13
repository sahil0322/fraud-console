"""End to end on a small generated file: training, leakage guards, rules baseline, loader."""
import contextlib, io, json, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd

from fraud import config as C
from fraud import synth, train
from fraud.scoring import Scorer
from fraud.features import FEATURES
from fraud.db import SqliteDb

ROOT = Path(__file__).resolve().parent.parent
_cache = {}


def trained():
    if "x" in _cache:
        return _cache["x"]
    tmp = Path(tempfile.mkdtemp(prefix="fraud-test-"))
    saved = {k: getattr(C, k) for k in ("ARTIFACT_DIR", "MODEL_PATH", "METRICS_PATH", "THRESHOLD_TABLE_PATH", "TEST_SCORED_PATH")}
    try:
        C.ARTIFACT_DIR = tmp
        C.MODEL_PATH, C.METRICS_PATH = tmp / "model.joblib", tmp / "metrics.json"
        C.THRESHOLD_TABLE_PATH, C.TEST_SCORED_PATH = tmp / "t.csv", tmp / "s.csv"
        csv = tmp / "synthetic_test.csv"
        synth.generate(80_000, fraud_rate=0.01, seed=3).to_csv(csv, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m = train.run(csv, None, C.COST_MISSED_FRAUD, C.COST_FALSE_ALERT)
        _cache["x"] = (tmp, csv, m, Scorer.load(tmp / "model.joblib"))
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
    return _cache["x"]


def test_model_beats_rules_only_baseline_on_the_later_test_window():
    _, _, m, _ = trained()
    c = m["test_comparison"]
    assert c["lr"]["pr_auc"] > c["rules"]["pr_auc"] and c["lr"]["recall_at_budget"] > c["rules"]["recall_at_budget"]


def test_test_window_is_strictly_later_than_train_and_validation():
    _, _, m, _ = trained()
    cuts = m["step_cuts"]
    assert cuts["train_end"] < cuts["val_end"] < cuts["max_step"]
    assert all(m["windows"][k]["fraud"] >= 20 for k in ("train", "val", "test"))


def test_no_label_or_leaky_column_is_a_model_feature():
    for bad in ("is_fraud", "isFraud", "isFlaggedFraud", "is_flagged_fraud", "txn_id", "name_orig", "name_dest"):
        assert bad not in FEATURES


def test_threshold_was_chosen_on_validation_and_report_has_the_headline_numbers():
    _, _, m, _ = trained()
    a = m["test_at_threshold"]
    assert 0 < m["threshold"] < 1.01 and 0 <= a["flagged_share"] <= 1
    assert 0 <= a["fraud_recall"] <= 1 and 0 <= a["alert_precision"] <= 1
    assert a["cost"] <= a["cost_if_nothing_flagged"]
    assert m["source"] == "synthetic"


def test_scored_alerts_have_one_to_three_reasons_and_flagged_rows_match_threshold():
    tmp, _, m, s = trained()
    out = pd.read_csv(tmp / "s.csv")
    assert len(out) == m["test_at_threshold"]["alerts"] and (out["risk"] >= s.threshold - 1e-12).all()
    for r in out["reasons"]:
        rs = json.loads(r)
        assert 1 <= len(rs) <= 3 and all(x["text"] for x in rs)


def test_loader_script_loads_alerts_and_rejects_a_different_csv_and_double_load():
    tmp, csv, m, _ = trained()
    db_file = tmp / "a.db"
    env_py = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
              f"from fraud import config as C; "
              f"C.ARTIFACT_DIR=__import__('pathlib').Path({str(tmp)!r}); "
              f"C.METRICS_PATH=C.ARTIFACT_DIR/'metrics.json'; C.TEST_SCORED_PATH=C.ARTIFACT_DIR/'s.csv'; "
              f"import os; os.environ['DATABASE_URL']={('sqlite:///' + db_file.as_posix())!r}; "
              f"sys.argv=['x','--csv',{str(csv)!r}]+sys.argv[1:]; "
              f"from scripts.load_alerts import main; main()")
    run = lambda *extra, **kw: subprocess.run([sys.executable, "-c", env_py + "", *extra], capture_output=True, text=True, cwd=ROOT)
    r = run()
    assert r.returncode == 0, r.stderr
    n = SqliteDb(str(db_file)).query("SELECT COUNT(*) AS n FROM alerts")[0]["n"]
    assert n == m["test_at_threshold"]["alerts"]
    assert run().returncode != 0                      # second load refuses without --reset
    assert run("--reset").returncode == 0
    other = tmp / "synthetic_other.csv"
    synth.generate(80_000, fraud_rate=0.01, seed=99).to_csv(other, index=False)
    bad = subprocess.run([sys.executable, "-c", env_py.replace(repr(str(csv)), repr(str(other)))],
                         capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode != 0 and "not the file" in (bad.stderr + bad.stdout)
