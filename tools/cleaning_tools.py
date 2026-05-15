from langchain.tools import tool
import pandas as pd
import numpy as np


GLOBAL_DF = None


def set_dataframe(df):
    """
    Store dataframe globally for tool access.
    """
    global GLOBAL_DF
    GLOBAL_DF = df.copy()


def get_dataframe():
    """
    Return the latest updated dataframe.
    """
    global GLOBAL_DF
    return GLOBAL_DF


@tool
def missing_value_tool(strategy: str = "median"):
    """
    Handle missing values using mean, median or mode strategy.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        df = GLOBAL_DF.copy()

        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns

        missing_before = int(df.isnull().sum().sum())

        # numeric columns
        for col in numeric_cols:

            if df[col].isnull().sum() == 0:
                continue

            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())

            else:
                df[col] = df[col].fillna(df[col].median())

        # categorical columns
        for col in categorical_cols:

            if df[col].isnull().sum() == 0:
                continue

            mode_values = df[col].mode()

            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values[0])

        missing_after = int(df.isnull().sum().sum())

        GLOBAL_DF = df

        return f"""
        Missing value handling completed successfully.

        Strategy Used: {strategy}

        Missing Values Before Cleaning: {missing_before}

        Missing Values After Cleaning: {missing_after}
        """

    except Exception as e:
        return f"Error while handling missing values: {str(e)}"


@tool
def duplicate_removal_tool(dummy: str = "remove"):
    """
    Remove duplicate rows from dataframe.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        before_rows = len(GLOBAL_DF)

        cleaned_df = GLOBAL_DF.drop_duplicates()

        after_rows = len(cleaned_df)

        removed_rows = before_rows - after_rows

        GLOBAL_DF = cleaned_df

        return f"""
        Duplicate removal completed successfully.

        Rows Before Cleaning: {before_rows}

        Rows After Cleaning: {after_rows}

        Duplicate Rows Removed: {removed_rows}
        """

    except Exception as e:
        return f"Error while removing duplicates: {str(e)}"


@tool
def outlier_detection_tool(column: str):
    """
    Detect outliers in a numeric column using IQR method.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        if column not in GLOBAL_DF.columns:
            return f"Column '{column}' does not exist in dataframe."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[column]):
            return f"Column '{column}' is not numeric."

        q1 = GLOBAL_DF[column].quantile(0.25)
        q3 = GLOBAL_DF[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = GLOBAL_DF[
            (GLOBAL_DF[column] < lower_bound)
            |
            (GLOBAL_DF[column] > upper_bound)
        ]

        outlier_count = len(outliers)

        return f"""
        Outlier detection completed successfully.

        Column: {column}

        Outlier Count: {outlier_count}

        Lower Bound: {lower_bound}

        Upper Bound: {upper_bound}
        """

    except Exception as e:
        return f"Error during outlier detection: {str(e)}"