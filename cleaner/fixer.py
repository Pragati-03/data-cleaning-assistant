import pandas as pd
from sklearn.impute import SimpleImputer

def fix_issues(df):
    df = df.copy()

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in cat_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    return df