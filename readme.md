# Data Cleaning Assistant

A tool that detects, explains, and fixes data quality issues in any CSV or Excel dataset — with a quality score, a change log, and a downloadable audit trail for every fix applied.

## Live Demo
[Try it here](https://data-cleaning-assistant-0.streamlit.app/)

---

## What it does

Most data scientists spend 80% of their time cleaning data. This tool automates that, without hiding what it's doing:

- **Detects missing values** — which columns, how many
- **Catches type issues** — numbers accidentally stored as strings
- **Flags outliers** — Isolation Forest (unsupervised ML)
- **Explains outliers** — SHAP values show *why* a row is suspicious
- **Flags duplicate rows** — and removes them on request
- **Scores data quality** — a single 0–100 score, before and after cleaning
- **Opt-in fixes** — choose which issues to fix and how, instead of one blanket auto-fix
- **Change log + report** — every fix is logged; the log is downloadable as a cleaning report alongside the cleaned CSV

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `Pandas` | Data loading and manipulation |
| `Scikit-learn` | Isolation Forest, SimpleImputer |
| `SHAP` | Explainability of outlier detection |
| `Plotly` | Interactive charts |
| `Streamlit` | Web UI |

---

## Project Structure
```
data-cleaning-assistant/
│
├── app.py                  # Streamlit UI
├── cleaner/
│   ├── __init__.py
│   ├── detector.py         # Issue detection + quality scoring
│   └── fixer.py            # Per-issue fixes + change log
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Pragati-03/data-cleaning-assistant
cd data-cleaning-assistant
pip install -r requirements.txt
streamlit run app.py
```

---

## ML Concepts Used

- **Isolation Forest** — unsupervised anomaly detection algorithm that isolates outliers by randomly partitioning data
- **SHAP (SHapley Additive exPlanations)** — explains model predictions by showing feature contributions
- **SimpleImputer** — fills missing values using median/mean (numeric) and mode (categorical) strategies

---

## About

Built as a portfolio project to demonstrate practical ML engineering and data preprocessing automation skills — with an emphasis on auditability (change logs, before/after scoring) over a black-box "auto fix" button.