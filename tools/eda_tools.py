from langchain.tools import tool
import pandas as pd


GLOBAL_DF = None


def set_dataframe(df):
    global GLOBAL_DF
    GLOBAL_DF = df


@tool
def descriptive_statistics_tool(dummy: str = "stats"):
    """Generate descriptive statistics for dataframe."""

    return GLOBAL_DF.describe(include='all').to_dict()


@tool
def missing_values_tool(dummy: str = "missing"):
    """Analyze missing values in dataframe."""

    return GLOBAL_DF.isnull().sum().to_dict()


@tool
def correlation_analysis_tool(dummy: str = "correlation"):
    """Perform correlation analysis."""

    return GLOBAL_DF.select_dtypes(include='number').corr().to_dict()


@tool
def categorical_summary_tool(column: str):
    """Generate categorical summary for a column."""

    return GLOBAL_DF[column].value_counts().to_dict()