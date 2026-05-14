import pandas as pd


def validate_dataset(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Dataset is empty")

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns)
    }