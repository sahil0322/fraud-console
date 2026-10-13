"""Alert queue, approve/block with mandatory comment, and dashboard statistics."""
import json

from fraud import service

from .helpers import add_alert, memory_db, raises


def test_queue_is_sorted_by_risk_highest_first_and_filters_by_status():
    db = memory_db()
    add_alert(db, 1, 0.80); add_alert(db, 2, 0.99); add_alert(db, 3, 0.90)
    assert [a["id"] for a in service.list_alerts(db)] == [2, 3, 1]
    service.act_on_alert(db, 2, "asha", "block", "confirmed")
    assert [a["id"] for a in service.list_alerts(db, "open")] == [3, 1]
    assert [a["id"] for a in service.list_alerts(db, "blocked")] == [2]
    assert len(service.list_alerts(db, "all")) == 3
    raises(ValueError, service.list_alerts, db, "nonsense")


def test_action_updates_status_and_stores_the_comment():
    db = memory_db(); add_alert(db, 1)
    out = service.act_on_alert(db, 1, " asha ", "approve", "  Customer confirmed by phone  ")
    assert out == {"alert_id": 1, "status": "approved"}
    d = service.get_alert(db, 1)
    assert d["status"] == "approved" and len(d["actions"]) == 1
    assert d["actions"][0]["analyst"] == "asha" and d["actions"][0]["comment"] == "Customer confirmed by phone"
    assert service.recent_actions(db)[0]["action"] == "approve"


def test_comment_and_analyst_are_mandatory_and_action_is_validated():
    db = memory_db(); add_alert(db, 1)
    for bad in ("", "   ", None):
        raises(service.InvalidAction, service.act_on_alert, db, 1, "asha", "approve", bad)
    raises(service.InvalidAction, service.act_on_alert, db, 1, "", "approve", "ok")
    raises(service.InvalidAction, service.act_on_alert, db, 1, "asha", "delete", "ok")
    assert service.get_alert(db, 1)["status"] == "open"
    assert db.query("SELECT COUNT(*) AS n FROM analyst_actions")[0]["n"] == 0


def test_alert_cannot_be_resolved_twice_and_missing_alert_is_reported():
    db = memory_db(); add_alert(db, 1)
    service.act_on_alert(db, 1, "asha", "approve", "ok")
    raises(service.AlertAlreadyResolved, service.act_on_alert, db, 1, "ben", "block", "changed my mind")
    raises(service.AlertNotFound, service.act_on_alert, db, 99, "asha", "block", "x")
    raises(service.AlertNotFound, service.get_alert, db, 99)
    assert db.query("SELECT COUNT(*) AS n FROM analyst_actions")[0]["n"] == 1


def test_database_rejects_blank_comment_even_if_the_service_is_bypassed():
    db = memory_db(); add_alert(db, 1)
    raises(Exception, db.query, "INSERT INTO analyst_actions (alert_id, analyst, action, comment, created_at) "
           "VALUES (1, 'asha', 'block', '   ', 'x')")


def test_failed_action_write_rolls_back_the_status_change():
    db = memory_db(); add_alert(db, 1)
    db.query("CREATE TRIGGER boom BEFORE INSERT ON analyst_actions BEGIN SELECT RAISE(ABORT, 'boom'); END")
    raises(Exception, service.act_on_alert, db, 1, "asha", "block", "x")
    assert service.get_alert(db, 1)["status"] == "open"


def test_detail_hides_the_ground_truth_label_and_parses_reasons():
    db = memory_db(); add_alert(db, 1, is_fraud=1)
    d = service.get_alert(db, 1)
    assert "is_fraud" not in d and isinstance(d["reasons"], list) and d["reasons"][0]["source"] == "model"


def test_stats_with_no_data_and_with_data():
    db = memory_db()
    e = service.stats(db, metrics_path="/nonexistent.json")
    assert e["alerts_by_status"] == {} and e["analyst_confirmed_precision"] is None
    assert e["label_precision"] is None and e["model_report"] is None
    add_alert(db, 1, is_fraud=1, step=10); add_alert(db, 2, is_fraud=0, step=10); add_alert(db, 3, is_fraud=1, step=11)
    service.act_on_alert(db, 1, "asha", "block", "fraud")
    service.act_on_alert(db, 2, "asha", "approve", "legit")
    s = service.stats(db, metrics_path="/nonexistent.json")
    assert s["alerts_by_status"] == {"blocked": 1, "approved": 1, "open": 1}
    assert s["analyst_resolved"] == 2 and s["analyst_confirmed_precision"] == 0.5
    assert s["label_alerts"] == 3 and abs(s["label_precision"] - 2 / 3) < 1e-9
    assert s["alerts_per_hour"] == [{"hour": 10, "alerts": 2}, {"hour": 11, "alerts": 1}]
