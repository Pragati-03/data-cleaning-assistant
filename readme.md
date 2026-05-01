# 🧹 Data Cleaning Assistant

An ML-powered tool that automatically detects, explains, and fixes data quality issues in any CSV dataset.

## 🚀 Live Demo
[Click here to try it live](#)

---

## 💡 What it does

Most data scientists spend 80% of their time cleaning data. This tool automates that:

- 🟡 **Detects missing values** — shows exactly which columns and how many
- 🔵 **Catches type issues** — finds numbers accidentally stored as strings
- 🔴 **Flags outliers** — using Isolation Forest (unsupervised ML)
- 🧠 **Explains outliers** — SHAP values show *why* a row is suspicious
- ✨ **Auto fixes** — imputes missing values, corrects types, exports clean CSV

---


## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Pandas` | Data loading and manipulation |
| `Scikit-learn` | Isolation Forest, SimpleImputer |
| `SHAP` | Explainability of outlier detection |
| `Plotly` | Interactive charts |
| `Streamlit` | Web UI |

---

## 📁 Project Structure
data-cleaning-assistant/
│
├── app.py                  # Streamlit UI
├── cleaner/
│   ├── init.py
│   ├── detector.py         # Issue detection logic
│   └── fixer.py            # Auto-fix logic
├── requirements.txt
└── README.md


---

## ⚙️ Run Locally

```bash
git clone https://github.com/pragati-03/data-cleaning-assistant
cd data-cleaning-assistant
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 ML Concepts Used

- **Isolation Forest** — unsupervised anomaly detection algorithm that isolates outliers by randomly partitioning data
- **SHAP (SHapley Additive exPlanations)** — explains model predictions by showing feature contributions
- **SimpleImputer** — fills missing values using median (numeric) and mode (categorical) strategies

---


## 🙋 About

Built as a portfolio project to demonstrate practical ML engineering and data preprocessing automation skills.