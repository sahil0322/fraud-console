"""What the console does with the database: list alerts, show one, approve or block, summarise.

An alert reaches a final state only through an analyst action, and the state change and
the action row are written in one transaction. Nothing is blocked automatically.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .config import METRICS_PATH

ACTIONS = {"approve": "approved", "block": "blocked"}


class AlertNotFound(Exception):
    pass


class AlertAlreadyResolved(Exception):
    pass


class InvalidAction(ValueError):
    pass


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _txt(v):
    """Timestamps come back as datetime from PostgreSQL and as text from SQLite."""
    return v.isoformat() if hasattr(v, "isoformat") else v


LIST_SQL = (
    "SELECT id, risk, status, step, type, amount, name_orig, name_dest, created_at FROM alerts "
    "WHERE (:status = 'all' OR status = :status) ORDER BY risk DESC, id ASC LIMIT :limit OFFSET :offset"
)


def list_alerts(db, status: str = "open", limit: int = 100, offset: int = 0) -> list[dict]:
    if status not in ("open", "approved", "blocked", "all"):
        raise ValueError("status must be open, approved, blocked or all")
    rows = db.query(LIST_SQL, {"status": status, "limit": int(limit), "offset": int(offset)})
    for r in rows:
        r["created_at"] = _txt(r["created_at"])
    return rows


def get_alert(db, alert_id: int) -> dict:
    rows = db.query(
        "SELECT id, risk, model_risk, rule_risk, reasons, status, step, type, amount, name_orig, "
        "old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest "
        "FROM alerts WHERE id = :id", {"id": alert_id})  # is_fraud is deliberately not selected
    if not rows:
        raise AlertNotFound(alert_id)
    alert = rows[0]
    alert["reasons"] = json.loads(alert["reasons"])
    alert["actions"] = db.query(
        "SELECT analyst, action, comment, created_at FROM analyst_actions "
        "WHERE alert_id = :id ORDER BY id", {"id": alert_id})
    for a in alert["actions"]:
        a["created_at"] = _txt(a["created_at"])
    return alert


def act_on_alert(db, alert_id: int, analyst: str, action: str, comment: str) -> dict:
    if action not in ACTIONS:
        raise InvalidAction("action must be approve or block")
    analyst, comment = (analyst or "").strip(), (comment or "").strip()
    if not analyst:
        raise InvalidAction("analyst name is required")
    if not comment:
        raise InvalidAction("a comment is required")
    with db.transaction() as tx:
        done = tx.query("UPDATE alerts SET status = :st WHERE id = :id AND status = 'open' RETURNING id",
                        {"st": ACTIONS[action], "id": alert_id})
        if not done:
            exists = tx.query("SELECT id FROM alerts WHERE id = :id", {"id": alert_id})
            raise AlertAlreadyResolved(alert_id) if exists else AlertNotFound(alert_id)
        tx.query("INSERT INTO analyst_actions (alert_id, analyst, action, comment, created_at) "
                 "VALUES (:aid, :who, :act, :c, :now) RETURNING id",
                 {"aid": alert_id, "who": analyst, "act": action, "c": comment, "now": utcnow()})
    return {"alert_id": alert_id, "status": ACTIONS[action]}


def recent_actions(db, limit: int = 50) -> list[dict]:
    rows = db.query(
        "SELECT x.created_at, x.analyst, x.action, x.comment, x.alert_id, a.risk "
        "FROM analyst_actions x JOIN alerts a ON a.id = x.alert_id ORDER BY x.id DESC LIMIT :limit",
        {"limit": int(limit)})
    for r in rows:
        r["created_at"] = _txt(r["created_at"])
    return rows


def stats(db, metrics_path: Path = METRICS_PATH) -> dict:
    by_status = {r["status"]: int(r["n"]) for r in db.query(
        "SELECT status, COUNT(*) AS n FROM alerts GROUP BY status")}
    per_hour = [{"hour": int(r["hour"]), "alerts": int(r["alerts"])} for r in db.query(
        "SELECT step AS hour, COUNT(*) AS alerts FROM alerts GROUP BY step ORDER BY step")]
    lab = db.query("SELECT COUNT(*) AS alerts, SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud "
                   "FROM alerts WHERE is_fraud IS NOT NULL")[0]
    approved, blocked = by_status.get("approved", 0), by_status.get("blocked", 0)
    resolved = approved + blocked
    n_label = int(lab["alerts"])
    report = None
    if Path(metrics_path).exists():
        report = json.loads(Path(metrics_path).read_text())
    return {
        "alerts_by_status": by_status,
        "alerts_per_hour": per_hour,
        "analyst_resolved": resolved,
        "analyst_confirmed_precision": (blocked / resolved) if resolved else None,
        "label_alerts": n_label,
        "label_precision": (int(lab["fraud"] or 0) / n_label) if n_label else None,
        "model_report": report,
    }
