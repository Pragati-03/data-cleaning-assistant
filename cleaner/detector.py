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

    return issues


def explain_outliers(df):
    from sklearn.ensemble import IsolationForest
    
    numeric_df = df.select_dtypes(include="number").dropna()
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(numeric_df)
    
    explainer = shap.Explainer(clf, numeric_df)
    shap_values = explainer(numeric_df)
    
    return shap_values, numeric_df