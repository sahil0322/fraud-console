"""Analyst console. Run from the project root:   streamlit run console/app.py

Reads DATABASE_URL (default: the local SQLite file made by scripts/load_alerts.py).
Three views: the alert queue with approve and block, a dashboard, and notes on the data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from fraud import config as C
from fraud import service
from fraud.db import connect

st.set_page_config(page_title="Fraud review console", page_icon=str(ROOT / "static" / "favicon.png"),
                   layout="wide")


@st.cache_resource
def get_db(url: str):
    return connect(url)


def money(v: float) -> str:
    return f"{v:,.2f}"


def alert_queue(db) -> None:
    st.header("Alert queue")
    st.caption("Highest risk first. Risk is a ranking score from 0 to 1, not a probability. "
               "Nothing is blocked until an analyst blocks it here.")
    who = st.sidebar.text_input("Your name", key="analyst")
    status = st.radio("Show", ["open", "approved", "blocked", "all"], horizontal=True)
    rows = service.list_alerts(db, status, limit=200)
    if not rows:
        st.info("No alerts with this status." if status != "open" else
                "No open alerts. Load some with: python -m scripts.load_alerts --csv <file>")
        return
    frame = pd.DataFrame(rows)[["id", "risk", "status", "type", "amount", "name_orig", "name_dest", "step"]]
    frame.columns = ["Alert", "Risk", "Status", "Type", "Amount", "Sender", "Recipient", "Hour"]
    st.dataframe(frame, hide_index=True, use_container_width=True,
                 column_config={"Risk": st.column_config.NumberColumn(format="%.3f"),
                                "Amount": st.column_config.NumberColumn(format="%.2f")})
    ids = [r["id"] for r in rows]
    chosen = st.selectbox("Open an alert", ids, format_func=lambda i: f"Alert {i}")
    alert_detail(db, int(chosen), who)


def alert_detail(db, alert_id: int, analyst: str) -> None:
    a = service.get_alert(db, alert_id)
    st.subheader(f"Alert {a['id']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Transaction**")
        st.write(f"{a['type']} of {money(a['amount'])} at hour {a['step']}")
        st.write(f"Sender {a['name_orig']}: balance {money(a['old_balance_orig'])} to {money(a['new_balance_orig'])}")
        st.write(f"Recipient {a['name_dest']}: balance {money(a['old_balance_dest'])} to {money(a['new_balance_dest'])}")
    with right:
        st.markdown("**Why it was flagged**")
        for r in a["reasons"]:
            st.write(f"{'Rule' if r['source'] == 'rule' else 'Model'}: {r['text']}")
        st.caption(f"Risk {a['risk']:.3f} (model {a['model_risk']:.3f}, rules {a['rule_risk']:.2f}). "
                   "Model reasons come from the largest logistic regression contributions.")
    if a["actions"]:
        st.markdown("**Decision history**")
        st.dataframe(pd.DataFrame(a["actions"]), hide_index=True, use_container_width=True)
    if a["status"] != "open":
        return
    with st.form(f"action-{alert_id}", clear_on_submit=True):
        comment = st.text_area("Comment (required)", max_chars=1000)
        c1, c2 = st.columns(2)
        approve = c1.form_submit_button("Approve")
        block = c2.form_submit_button("Block")
    if approve or block:
        action = "approve" if approve else "block"
        try:
            service.act_on_alert(db, alert_id, analyst, action, comment)
        except service.InvalidAction as e:
            st.error(str(e))
        except service.AlertAlreadyResolved:
            st.error("Someone already resolved this alert. Reload the queue.")
        else:
            st.success(f"Alert {alert_id} {service.ACTIONS[action]}.")
            st.rerun()


def pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def dashboard(db) -> None:
    st.header("Dashboard")
    s = service.stats(db)
    by = s["alerts_by_status"]
    c = st.columns(4)
    c[0].metric("Open alerts", by.get("open", 0))
    c[1].metric("Approved", by.get("approved", 0))
    c[2].metric("Blocked", by.get("blocked", 0))
    c[3].metric("Blocked share of resolved", pct(s["analyst_confirmed_precision"]),
                help="Analyst view: blocked divided by approved plus blocked. Needs resolved alerts.")
    st.metric("Alert precision against the dataset labels", pct(s["label_precision"]),
              help="Share of alerts that are real fraud in the dataset. Labels are hidden from the analyst view.")
    if s["alerts_per_hour"]:
        st.subheader("Alerts per hour of the test period")
        st.bar_chart(pd.DataFrame(s["alerts_per_hour"]).set_index("hour")["alerts"])

    m = s["model_report"]
    if not m:
        st.info("No model report found. Run: python -m fraud.train --csv <file>")
        return
    if m["source"] == "synthetic":
        st.warning("These numbers come from synthetic data made to look like PaySim. "
                   "They show the pipeline works, not how it performs on real transactions.")
    st.subheader("Logistic regression against a rules-only baseline")
    b = int(round(m["review_budget_frac"] * 100))
    names = {"rules": "Rules only", "lr": "Logistic regression", "lr+rules": "Logistic regression plus rules"}
    comp = pd.DataFrame([{"Approach": names[k], "PR-AUC": v["pr_auc"],
                          f"Fraud caught reviewing the top {b}%": v["recall_at_budget"],
                          f"Alert precision at the top {b}%": v["precision_at_budget"]}
                         for k, v in m["test_comparison"].items()])
    st.dataframe(comp, hide_index=True, use_container_width=True,
                 column_config={col: st.column_config.NumberColumn(format="%.3f") for col in comp.columns[1:]})
    st.caption("Measured on the last 25% of the timeline, which the model never saw. "
               "The threshold was chosen on a separate middle window.")

    st.subheader("Review workload against fraud caught")
    a, r = m["test_at_threshold"], m["test_rules_only_at_natural_point"]
    st.write(f"With the chosen threshold ({m['threshold']:.2f}), analysts review {a['flagged_share']:.2%} of "
             f"transactions and catch {a['fraud_recall']:.1%} of fraud, with {a['alert_precision']:.1%} of alerts "
             f"being real fraud. Rules alone would review {r['flagged_share']:.2%} and catch {r['fraud_recall']:.1%}.")
    cost = m["cost_assumptions"]
    st.write(f"Cost model: a missed fraud costs {cost['missed_fraud']:g} and a false alert {cost['false_alert']:g}. "
             f"Total cost {a['cost']:,.0f}, against {a['cost_if_nothing_flagged']:,.0f} if nothing is flagged "
             f"and {a['cost_if_everything_flagged']:,.0f} if everything is.")
    if C.THRESHOLD_TABLE_PATH.exists():
        t = pd.read_csv(C.THRESHOLD_TABLE_PATH)
        t = t[(t["alerts"] > 0)].copy()
        t["Share reviewed"] = t["alert_rate"]
        chart = t.set_index("Share reviewed")[["recall"]].rename(columns={"recall": "Fraud caught"})
        st.line_chart(chart)
        st.caption("Validation window. Each point is one threshold.")


def notes(db) -> None:
    st.header("About the data and limits")
    s = service.stats(db)
    m = s["model_report"] or {}
    st.write(f"Data source: **{m.get('source', 'unknown')}**. Model version: {m.get('model_version', 'none')}.")
    st.markdown(
        "- Fraud rate and fraud patterns in simulated data differ from real payments. A strong score here "
        "does not carry over to production.\n"
        "- The data has no customer identity beyond account ids, no device or location, and a fixed "
        "simulation process, so the model may learn the simulator's habits.\n"
        "- Balance fields can leak the outcome in PaySim. They are kept as features because a payment "
        "system sees them, but they are the first thing to question.\n"
        "- The provider's own isFlaggedFraud column is dropped, since it is computed from the label.\n"
        "- The risk score is a ranking, not a probability.")


def main() -> None:
    db = get_db(os.environ.get("DATABASE_URL", C.DEFAULT_DB_URL))
    try:
        db.query("SELECT 1 AS ok")
        db.query("SELECT COUNT(*) AS n FROM alerts")
    except Exception:
        st.error("The alerts database is empty or missing. Run train and load_alerts first (see README).")
        st.stop()
    page = st.sidebar.radio("View", ["Alert queue", "Dashboard", "About the data"])
    {"Alert queue": alert_queue, "Dashboard": dashboard, "About the data": notes}[page](db)


main()
