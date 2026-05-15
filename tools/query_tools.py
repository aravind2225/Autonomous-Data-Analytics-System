from langchain.tools import tool
import pandas as pd


GLOBAL_DF = None


def set_dataframe(df):

    global GLOBAL_DF
    GLOBAL_DF = df.copy()


def get_dataframe():

    global GLOBAL_DF
    return GLOBAL_DF


@tool
def top_records_tool(column: str):
    """
    Return top 5 records from numeric column.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        if column not in GLOBAL_DF.columns:
            return f"Column '{column}' not found."

        if not pd.api.types.is_numeric_dtype(
            GLOBAL_DF[column]
        ):
            return f"Column '{column}' is not numeric."

        top = GLOBAL_DF.nlargest(5, column)

        return f"""
        Top 5 records for column '{column}':

        {top.to_string()}
        """

    except Exception as e:

        return f"Error in top_records_tool: {str(e)}"


@tool
def grouping_tool(columns: str):
    """
    Group dataframe using categorical and numeric columns.

    Input format:
    categorical_column,numeric_column
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        group_col, agg_col = columns.split(",")

        group_col = group_col.strip()
        agg_col = agg_col.strip()

        if group_col not in GLOBAL_DF.columns:
            return f"Column '{group_col}' not found."

        if agg_col not in GLOBAL_DF.columns:
            return f"Column '{agg_col}' not found."

        result = (
            GLOBAL_DF.groupby(group_col)[agg_col]
            .mean()
            .sort_values(ascending=False)
        )

        return f"""
        Grouped analysis completed.

        Average {agg_col} grouped by {group_col}:

        {result.to_string()}
        """

    except Exception as e:

        return f"Error in grouping_tool: {str(e)}"


@tool
def dataframe_summary_tool(dummy: str = "summary"):
    """
    Generate dataframe summary.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        return f"""
        Dataframe Summary

        Shape:
        {GLOBAL_DF.shape}

        Columns:
        {list(GLOBAL_DF.columns)}

        Data Types:
        {GLOBAL_DF.dtypes.astype(str).to_dict()}
        """

    except Exception as e:

        return f"Error in dataframe_summary_tool: {str(e)}"