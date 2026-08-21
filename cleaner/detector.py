import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

def detect_issues(df):
    issues = {}

    # Missing values
    missing = df.isnull().sum()
    issues["missing_values"] = missing[missing > 0].to_dict()

    # Wrong data types (numbers stored as strings)
    type_issues = {}
    for col in df.columns:
        if df[col].dtype == object:
            try:
                pd.to_numeric(df[col])
                type_issues[col] = "Looks numeric but stored as string"
            except:
                pass
    issues["type_issues"] = type_issues

    # Outliers using Isolation Forest
    numeric_df = df.select_dtypes(include="number").dropna()
    if not numeric_df.empty:
        clf = IsolationForest(contamination=0.05, random_state=42)
        preds = clf.fit_predict(numeric_df)
        outlier_count = (preds == -1).sum()
        issues["outliers"] = {"count": int(outlier_count)}
    else:
        issues["outliers"] = {"count": 0}

    issues["duplicates"] = {"count": int(df.duplicated().sum())}
    return issues


def explain_outliers(df):
    from sklearn.ensemble import IsolationForest

    numeric_df = df.select_dtypes(include="number").dropna()
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(numeric_df)

    # IsolationForest itself isn't callable — SHAP needs a function that
    # takes an array and returns scores. decision_function gives the
    # anomaly score (lower = more anomalous), which is what we want to explain.
    explainer = shap.Explainer(clf.decision_function, numeric_df)
    shap_values = explainer(numeric_df)

    return shap_values, numeric_df

def calculate_quality_score(df, issues):
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0.0

    score = 100

    # Penalize missing values
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (df.isnull().sum().sum() / total_cells) * 100
    score -= min(missing_pct * 2, 40)  # max 40 point penalty

    # Penalize outliers
    outlier_count = issues.get("outliers", {}).get("count", 0)
    outlier_pct = (outlier_count / df.shape[0]) * 100
    score -= min(outlier_pct * 2, 30)  # max 30 point penalty

    # Penalize type issues
    score -= min(len(issues.get("type_issues", {})) * 5, 15)  # max 15 point penalty

    # Penalize duplicates
    dup_count = issues.get("duplicates", {}).get("count", 0)
    dup_pct = (dup_count / df.shape[0]) * 100
    score -= min(dup_pct * 2, 15)  # max 15 point penalty

    return round(max(score, 0), 1)