from langchain.tools import tool
import pandas as pd
import numpy as np


GLOBAL_DF = None


def set_dataframe(df):
    """
    Store dataframe globally for EDA tools.
    """

    global GLOBAL_DF
    GLOBAL_DF = df.copy()


def get_dataframe():
    """
    Return current dataframe.
    """

    global GLOBAL_DF
    return GLOBAL_DF


@tool
def descriptive_statistics_tool(dummy: str = "stats"):
    """
    Generate descriptive statistics for dataframe.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        stats = GLOBAL_DF.describe(include='all')

        summary = stats.to_string()

        return f"""
        Descriptive statistics generated successfully.

        Dataset Shape: {GLOBAL_DF.shape}

        Statistics Summary:

        {summary}
        """

    except Exception as e:
        return f"Error generating descriptive statistics: {str(e)}"


@tool
def missing_values_tool(dummy: str = "missing"):
    """
    Analyze missing values in dataframe.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        missing = GLOBAL_DF.isnull().sum()

        missing = missing[missing > 0]

        if missing.empty:
            return "No missing values found in dataset."

        result = missing.to_string()

        return f"""
        Missing value analysis completed.

        Columns With Missing Values:

        {result}
        """

    except Exception as e:
        return f"Error analyzing missing values: {str(e)}"


@tool
def correlation_analysis_tool(dummy: str = "correlation"):
    """
    Perform correlation analysis on numeric columns.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        numeric_df = GLOBAL_DF.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            return "Not enough numeric columns for correlation analysis."

        correlation_matrix = numeric_df.corr()

        result = correlation_matrix.to_string()

        return f"""
        Correlation analysis completed successfully.

        Correlation Matrix:

        {result}
        """

    except Exception as e:
        return f"Error during correlation analysis: {str(e)}"


@tool
def categorical_summary_tool(column: str):
    """
    Generate categorical summary for a column.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        if column not in GLOBAL_DF.columns:
            return f"Column '{column}' not found."

        value_counts = GLOBAL_DF[column].value_counts()

        result = value_counts.to_string()

        return f"""
        Categorical summary generated successfully.

        Column: {column}

        Value Counts:

        {result}
        """

    except Exception as e:
        return f"Error generating categorical summary: {str(e)}"