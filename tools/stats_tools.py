from langchain.tools import tool
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import zscore
import pandas as pd
import numpy as np


GLOBAL_DF = None


def set_dataframe(df):
    """
    Store dataframe globally for statistical tools.
    """

    global GLOBAL_DF
    GLOBAL_DF = df.copy()


def get_dataframe():
    """
    Return dataframe.
    """

    global GLOBAL_DF
    return GLOBAL_DF


@tool
def t_test_tool(columns: str):
    """
    Perform independent t-test between two numeric columns.

    Input format:
    column1,column2
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        col1, col2 = columns.split(",")

        col1 = col1.strip()
        col2 = col2.strip()

        if col1 not in GLOBAL_DF.columns:
            return f"Column '{col1}' not found."

        if col2 not in GLOBAL_DF.columns:
            return f"Column '{col2}' not found."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[col1]):
            return f"Column '{col1}' is not numeric."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[col2]):
            return f"Column '{col2}' is not numeric."

        data1 = GLOBAL_DF[col1].dropna()
        data2 = GLOBAL_DF[col2].dropna()

        if len(data1) < 2 or len(data2) < 2:
            return "Not enough data for t-test."

        stat, p = ttest_ind(data1, data2)

        interpretation = (
            "Statistically significant difference detected."
            if p < 0.05
            else "No statistically significant difference detected."
        )

        return f"""
        T-Test completed successfully.

        Column 1: {col1}
        Column 2: {col2}

        T-Statistic: {round(float(stat), 4)}

        P-Value: {round(float(p), 4)}

        Interpretation:
        {interpretation}
        """

    except Exception as e:
        return f"Error during t-test: {str(e)}"


@tool
def correlation_test_tool(columns: str):
    """
    Perform Pearson correlation test.

    Input format:
    column1,column2
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        col1, col2 = columns.split(",")

        col1 = col1.strip()
        col2 = col2.strip()

        if col1 not in GLOBAL_DF.columns:
            return f"Column '{col1}' not found."

        if col2 not in GLOBAL_DF.columns:
            return f"Column '{col2}' not found."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[col1]):
            return f"Column '{col1}' is not numeric."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[col2]):
            return f"Column '{col2}' is not numeric."

        df = GLOBAL_DF[[col1, col2]].dropna()

        if len(df) < 3:
            return "Not enough data for correlation analysis."

        corr, p = pearsonr(
            df[col1],
            df[col2]
        )

        return f"""
        Pearson correlation analysis completed.

        Column 1: {col1}
        Column 2: {col2}

        Correlation Coefficient: {round(float(corr), 4)}

        P-Value: {round(float(p), 4)}
        """

    except Exception as e:
        return f"Error during correlation analysis: {str(e)}"


@tool
def anomaly_detection_tool(column: str):
    """
    Detect anomalies using z-score method.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        if column not in GLOBAL_DF.columns:
            return f"Column '{column}' not found."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[column]):
            return f"Column '{column}' is not numeric."

        series = GLOBAL_DF[column].dropna()

        if len(series) < 3:
            return "Not enough data for anomaly detection."

        scores = np.abs(zscore(series))

        anomalies = series[scores > 3]

        return f"""
        Anomaly detection completed successfully.

        Column: {column}

        Anomaly Count: {len(anomalies)}
        """

    except Exception as e:
        return f"Error during anomaly detection: {str(e)}"