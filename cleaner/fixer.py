import pandas as pd
from sklearn.impute import SimpleImputer


def fix_issues_with_log(df):
    """
    Cleans df and returns (cleaned_df, imputation_log) where imputation_log
    is {column_name: strategy_used} — "mean"/"median" for numeric columns
    (chosen by skew), "mode" for categorical columns. Matches the contract
    app.py expects: pd.DataFrame.from_dict(imputation_log, orient="index", ...).
    """
    df = df.copy()
    log = {}

    # Fix numeric-looking text columns first, so they're treated as numeric below.
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Smart imputation for numeric columns: median for skewed data, mean otherwise.
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isnull().sum() == 0:
            continue
        skew = df[col].skew()
        if pd.notna(skew) and abs(skew) > 1:
            strategy = "median"
            fill_value = df[col].median()
        else:
            strategy = "mean"
            fill_value = df[col].mean()
        df[col] = df[col].fillna(fill_value)
        log[col] = strategy

    # Mode imputation for remaining categorical columns.
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().sum() == 0:
            continue
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
            log[col] = "mode"

    # Drop duplicate rows — previously counted in the UI but never removed.
    df = df.drop_duplicates(keep="first")

    if not log:
        log["(none)"] = "no missing values found"

    return df, log


def fix_issues(df):
    """Simple version with no per-column log — kept for compatibility."""
    cleaned_df, _ = fix_issues_with_log(df)
    return cleaned_df