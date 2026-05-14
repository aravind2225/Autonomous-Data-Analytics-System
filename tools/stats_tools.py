from langchain.tools import tool
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import zscore
import numpy as np


GLOBAL_DF = None


def set_dataframe(df):
    global GLOBAL_DF
    GLOBAL_DF = df


@tool
def t_test_tool(columns: str):
    """Perform t-test between two numeric columns."""

    col1, col2 = columns.split(",")

    stat, p = ttest_ind(
        GLOBAL_DF[col1.strip()].dropna(),
        GLOBAL_DF[col2.strip()].dropna()
    )

    return {
        "t_statistic": float(stat),
        "p_value": float(p)
    }


@tool
def correlation_test_tool(columns: str):
    """Perform Pearson correlation test."""

    col1, col2 = columns.split(",")

    corr, p = pearsonr(
        GLOBAL_DF[col1.strip()],
        GLOBAL_DF[col2.strip()]
    )

    return {
        "correlation": float(corr),
        "p_value": float(p)
    }


@tool
def anomaly_detection_tool(column: str):
    """Detect anomalies using z-score method."""

    scores = np.abs(zscore(GLOBAL_DF[column]))

    anomalies = GLOBAL_DF[scores > 3]

    return {
        "anomaly_count": len(anomalies)
    }