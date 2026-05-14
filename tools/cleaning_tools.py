from langchain.tools import tool
import pandas as pd
import numpy as np

GLOBAL_DF = None


def set_dataframe(df):
    global GLOBAL_DF
    GLOBAL_DF = df


@tool
def missing_value_tool(strategy: str = "median"):
    """Handle missing values using mean, median or mode strategy."""

    df = GLOBAL_DF.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


@tool
def duplicate_removal_tool(dummy: str = "remove"):
    """Remove duplicate rows from dataframe."""

    return GLOBAL_DF.drop_duplicates()


@tool
def outlier_detection_tool(column: str):
    """Detect outliers using IQR method."""

    q1 = GLOBAL_DF[column].quantile(0.25)
    q3 = GLOBAL_DF[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = GLOBAL_DF[
        (GLOBAL_DF[column] < lower)
        |
        (GLOBAL_DF[column] > upper)
    ]

    return {
        "column": column,
        "outlier_count": len(outliers)
    }