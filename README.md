# Fraud Review Console

Screening for mobile-money transaction fraud, with a review queue for analysts.

A logistic regression and three readable rules score each transaction. Flagged transactions go into a queue where an analyst approves or blocks them and records a comment. The project is built around the questions an analyst would ask of such a system: are the features free of look-ahead, does the model beat a simple rules baseline, how many transactions must be reviewed to catch how much fraud, and where should the results not be trusted.

**Author:** [sahil0322](https://github.com/sahil0322)

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [The console](#the-console)
- [Method](#method)
- [Data](#data)
- [Limitations](#limitations)
- [Testing](#testing)
- [Project layout](#project-layout)
- [License](#license)

## Results

### Application screenshots

**Alert queue**

![Fraud alert queue](screenshots/alert-queue.png)

**Dashboard**

![Fraud detection dashboard](screenshots/dashboard.png)

### PaySim evaluation

The following results were obtained by running the training pipeline on the downloaded PaySim dataset containing 6,362,620 transactions. The model was evaluated on a later-in-time test window of 193,939 transactions, including 2,062 fraud cases.

| Approach | PR-AUC | Fraud caught in top 1% reviewed | Precision in top 1% reviewed |
|---|---:|---:|---:|
| Rules only | 0.048 | 4.8% | 5.2% |
| Logistic regression | 0.962 | 85.3% | 90.7% |
| Logistic regression plus rules | 0.958 | 85.3% | 90.7% |

At the selected threshold of 0.720, the combined screening pipeline reviewed 1.29% of transactions, caught 99.3% of fraud, and achieved 82.2% alert precision in this evaluation run. Rules alone reviewed 21.79% of transactions, caught 99.6% of fraud, and achieved 4.9% precision.

These results describe performance on the PaySim simulator dataset only. They are not evidence of equivalent performance on real banking transactions.

### Synthetic-data baseline

The figures below come from the synthetic dataset in `fraud/synth.py` (400,000 transactions, 0.25% fraud), not from PaySim. They show what the pipeline reports. They are not evidence of real-world performance. See [Data](#data) and [Limitations](#limitations).

The model was fitted on the first 60% of the timeline, the threshold was chosen on the next 15%, and everything below was measured on the final 25% (102,431 transactions, 230 of them fraud).

| Approach | PR-AUC | Fraud caught when reviewing the top 1% | Alert precision at the top 1% |
|---|---:|---:|---:|
| Rules only | 0.012 | 10.4% | 2.3% |
| Logistic regression | 0.396 | 68.7% | 15.4% |
| Logistic regression plus rules | 0.394 | 68.7% | 15.4% |

With the threshold chosen by cost (a missed fraud is assumed to cost 100 times a false alert), analysts review 2.15% of transactions, catch 77.4% of the fraud, and 8.1% of alerts are real fraud. Rules alone, flagging every row where any rule fires, review 3.41% and catch 37.0%.

Adding the rules to the model did not improve on the model alone in this run.

## How it works

```text
CSV  ->  load and check columns  ->  SQL window features  ->  feature table
                                                                    |
              rules (3)  ------------------------------------+      |
              logistic regression  ---------------------------+----> risk = max(rule risk, model risk)
                                                                    |
                              risk >= threshold  ->  alerts table  ->  analyst queue
                                                                    |
                                          approve or block + comment  ->  analyst_actions table
```

1. **Load.** The loader checks that the expected PaySim columns exist and drops `isFlaggedFraud`, which is derived from the fraud label.
2. **History features in SQL.** One query with window functions computes, for every transaction, what the sender and recipient did before it.
3. **Score.** Three rules and a logistic regression each produce a risk. The transaction's risk is the higher of the two.
4. **Choose a threshold.** The threshold with the lowest total cost on a validation window is used.
5. **Review.** Transactions at or above the threshold become alerts. Nothing is blocked automatically. An alert is closed only when an analyst approves or blocks it and writes a comment.

## Quick start

Requires Python 3.10 or newer. The window functions need SQLite 3.28 or newer, which current Python releases bundle.

```bash
pip install -r requirements.txt

# 1. Get data. Either generate the synthetic stand-in...
python -m fraud.synth
# ...or download PaySim from Kaggle and use its path below instead.

# 2. Build features, train, compare, choose the threshold, write the report
python -m fraud.train --csv data/synthetic_paysim.csv

# 3. Load the flagged transactions into the database
python -m scripts.load_alerts --csv data/synthetic_paysim.csv

# 4. Open the console
streamlit run console/app.py
```

`load_alerts` refuses a CSV that is not the file the model was trained on. Add `--reset` to replace alerts that are already loaded.

### PostgreSQL (optional)

SQLite is the default and needs no setup. To store alerts in PostgreSQL:

```bash
docker compose up -d
export DATABASE_URL=postgresql://fraud:fraud@localhost:5432/fraud
python -m scripts.load_alerts --csv data/synthetic_paysim.csv
streamlit run console/app.py
```

The schema is two tables. `sql/schema.postgres.sql` and `sql/schema.sqlite.sql` define the same columns.

### Options for training

| Option | Default | Meaning |
|---|---|---|
| `--csv` | required | Path to the transactions CSV |
| `--cost-missed` | 1000 | Cost of a missed fraud |
| `--cost-false` | 10 | Cost of a false alert |
| `--nrows` | all | Read only the first N rows, for a quick run |

Only the ratio of the two costs matters. Other settings (window length, rule thresholds, split points, review budget) are in `fraud/config.py`.

## The console

- **Alert queue.** Alerts sorted by risk, highest first, filterable by status. Opening one shows the transaction, the balances on both sides, and up to three reasons for the flag.
- **Approve or block.** Both require a comment and an analyst name. The status change and the recorded action are written in one database transaction, and an alert that has already been resolved cannot be resolved again.
- **Dashboard.** Counts of open, approved and blocked alerts, alert precision, alerts per hour, the rules-versus-model comparison, and the review-workload-against-fraud-caught curve.
- **About the data.** A short statement of the data source and its limits.

Each alert lists its reasons in plain words. Fired rules come first. After that come the logistic regression terms that raised the score most, computed as coefficient times standardised feature value. The ground-truth label is stored for the dashboard's precision figure only and is never shown on an alert.

## Method

### Features

| Group | Features |
|---|---|
| Transaction | log amount, hour of day, transaction type |
| Balance | amount as a share of sender balance (capped at 2), account-drain flag, sender balance was zero, signed log gap between expected and recorded sender balance, same for the recipient, recipient balance was zero |
| History (SQL) | sender transactions in the previous 6 hours, log amount sent in that window, amount relative to the sender's own earlier average, sender has any history, first payment from this sender to this recipient |

History features count only strictly earlier hours. PaySim's clock is one hour, so the order of transactions inside an hour is unknown and same-hour rows are not counted. The window-function query is in `fraud/history.py` and was checked to give identical output on SQLite and PostgreSQL 16.

### Rules

| Rule | Fires when | Risk |
|---|---|---:|
| Drain account | A transfer or cash-out takes at least 95% of the sender's balance | 0.70 |
| New recipient, large | A transfer or cash-out of at least 200,000 goes to an account whose balance was zero | 0.60 |
| High velocity | The sender made 3 or more transactions in the previous 6 hours | 0.55 |

These are round numbers fixed in advance. They were not tuned.

### Model

Logistic regression with standardised numeric features, one-hot transaction type and balanced class weights. Because of the class weights the output is a ranking score between 0 and 1, not a probability. A score of 0.9 does not mean a 90% chance of fraud.

### Validation

The split is by time, never random:

| Window | Share of the timeline | Used for |
|---|---|---|
| Train | first 60% | fitting the model |
| Validation | next 15% | choosing the threshold |
| Test | last 25% | reporting, with no choices made on it |

A random split would let the model see later transactions from the same accounts and inflate every figure. Accuracy is not reported anywhere, since a model that flags nothing is over 99% accurate on data like this.

### Metrics

- **PR-AUC**, which suits rare events better than ROC-AUC.
- **Recall and precision at a review budget.** Rank the test transactions by risk, review the top 1%, and count how much fraud that catches and how many alerts are real. Ties are broken with a fixed random seed so a rules baseline with many equal scores is neither helped nor hurt by row order.
- **Cost at the chosen threshold**, compared with flagging nothing and with flagging everything.

## Data

The project is written for the PaySim dataset (Lopez-Rojas, Elmir and Axelsson, "PaySim: A financial mobile money simulator for fraud detection", 2016), available on Kaggle as "Synthetic Financial Datasets For Fraud Detection". It is not included in this repository.

`fraud/synth.py` generates a stand-in with PaySim's column names so the project runs without the download. It includes repeat senders, legitimate customers who empty their accounts, balance mismatches on legitimate rows, and a share of harder fraud that grows over time. It is a test fixture and its results must not be quoted as PaySim results. The training report labels the data source and the console shows a warning when the data is synthetic.

## Limitations

**The included data is synthetic.** It was written for this project, so the rules baseline and the history features behave in ways that depend on choices made in the generator. Treat the results table as a demonstration of the pipeline.

**Real PaySim differs in ways that matter.**

- Most sender accounts in PaySim appear only once, so the history features will be empty for nearly every row and add little. `fraud.train` prints the share of distinct senders so this can be checked.
- Fraud in PaySim is limited to transfers and cash-outs and often empties the account. Models score very high on it. A near-perfect score there says more about the simulator than about the detector.
- The loader checks the expected column names and lists any that are missing.

**Leakage.**

- `isFlaggedFraud` is computed from the fraud label and is dropped on load.
- Balance fields, and the gaps between expected and recorded balances, can reveal the outcome in simulated data. They are kept because a payment system sees them, but they are the first features to question.
- History features use earlier hours only.

**The threshold depends on assumed costs.** The 1000 to 10 ratio is an assumption. A different ratio gives a different threshold.

**Rule thresholds were not tuned.** The baseline is simple on purpose. A tuned rule set would be a stronger baseline.

**Small counts.** The test window holds 230 fraud cases, so differences of a few points between approaches are within noise. No confidence intervals are reported.

**One model, one dataset.** Nothing here shows that the approach holds on other data or over a longer period. There is no retraining and no drift monitoring.

**Batch only.** The model scores a file and alerts are loaded afterwards. There is no live scoring service. The console has no login and the analyst name is typed in, so it should not be exposed to a network as it stands.

## Testing

```bash
pytest
```

The tests cover the rules and feature definitions, the loader's column check and label handling, the window-function history features against hand-computed cases, the time split, the recall-at-budget and cost arithmetic, a full train-and-evaluate run on a small generated file, the alert loader, and the alert queue and approve-or-block logic including the mandatory comment and rollback on failure.

## Project layout

```text
fraud/
  synth.py        synthetic stand-in with PaySim's column names
  data.py         loader: column check, drops isFlaggedFraud
  history.py      SQL window-function features
  features.py     feature engineering
  rules.py        the three rules
  train.py        split, fit, compare, choose threshold, write report
  metrics.py      recall at a review budget, cost table
  scoring.py      risk score and reasons
  service.py      alert queue, approve and block, statistics
  db.py           SQLite and PostgreSQL access
  config.py       every tunable setting
scripts/
  load_alerts.py  load flagged transactions into the database
console/
  app.py          Streamlit console
sql/              two-table schema for SQLite and PostgreSQL
tests/
static/           favicon
```

## License

This project's code is licensed under the [MIT License](LICENSE).

The PaySim dataset is not included in this repository and remains subject to its own terms and conditions.
