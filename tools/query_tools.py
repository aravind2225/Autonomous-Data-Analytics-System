from langchain.tools import tool

GLOBAL_DF = None


def set_dataframe(df):
    global GLOBAL_DF
    GLOBAL_DF = df


@tool
def top_records_tool(column: str):
    """Return top records from numeric column."""

    top = GLOBAL_DF.nlargest(5, column)

    return top.to_dict()


@tool
def grouping_tool(columns: str):
    """Group dataframe using categorical and numeric columns."""

    group_col, agg_col = columns.split(",")

    result = GLOBAL_DF.groupby(group_col.strip())[agg_col.strip()].mean()

    return result.to_dict()


@tool
def dataframe_summary_tool(dummy: str = "summary"):
    """Generate dataframe summary."""

    return {
        "shape": GLOBAL_DF.shape,
        "columns": list(GLOBAL_DF.columns)
    }