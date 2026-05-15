from langchain.tools import tool
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np


GLOBAL_DF = None


def set_dataframe(df):
    """
    Store dataframe globally for visualization tools.
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
def histogram_tool(column: str):
    """
    Generate histogram for numeric column.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        if column not in GLOBAL_DF.columns:
            return f"Column '{column}' not found."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[column]):
            return f"Column '{column}' is not numeric."

        fig = px.histogram(
            GLOBAL_DF,
            x=column,
            title=f"Histogram of {column}"
        )

        return {
            "type": "histogram",
            "column": column,
            "figure": fig
        }

    except Exception as e:
        return f"Error generating histogram: {str(e)}"


@tool
def scatter_tool(columns: str):
    """
    Generate scatter plot using two numeric columns.

    Input format:
    column1,column2
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        x, y = columns.split(",")

        x = x.strip()
        y = y.strip()

        if x not in GLOBAL_DF.columns:
            return f"Column '{x}' not found."

        if y not in GLOBAL_DF.columns:
            return f"Column '{y}' not found."

        fig = px.scatter(
            GLOBAL_DF,
            x=x,
            y=y,
            title=f"{x} vs {y}"
        )

        return {
            "type": "scatter",
            "x": x,
            "y": y,
            "figure": fig
        }

    except Exception as e:
        return f"Error generating scatter plot: {str(e)}"


@tool
def line_chart_tool(columns: str):
    """
    Generate line chart.

    Input format:
    column1,column2
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        x, y = columns.split(",")

        x = x.strip()
        y = y.strip()

        if x not in GLOBAL_DF.columns:
            return f"Column '{x}' not found."

        if y not in GLOBAL_DF.columns:
            return f"Column '{y}' not found."

        fig = px.line(
            GLOBAL_DF,
            x=x,
            y=y,
            title=f"{y} over {x}"
        )

        return {
            "type": "line",
            "x": x,
            "y": y,
            "figure": fig
        }

    except Exception as e:
        return f"Error generating line chart: {str(e)}"


@tool
def bar_chart_tool(columns: str):
    """
    Generate bar chart.

    Input format:
    categorical_column,numeric_column
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        x, y = columns.split(",")

        x = x.strip()
        y = y.strip()

        if x not in GLOBAL_DF.columns:
            return f"Column '{x}' not found."

        if y not in GLOBAL_DF.columns:
            return f"Column '{y}' not found."

        fig = px.bar(
            GLOBAL_DF,
            x=x,
            y=y,
            title=f"{y} by {x}"
        )

        return {
            "type": "bar",
            "x": x,
            "y": y,
            "figure": fig
        }

    except Exception as e:
        return f"Error generating bar chart: {str(e)}"


@tool
def pie_chart_tool(columns: str):
    """
    Generate pie chart.

    Input format:
    categorical_column,numeric_column
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        names, values = columns.split(",")

        names = names.strip()
        values = values.strip()

        if names not in GLOBAL_DF.columns:
            return f"Column '{names}' not found."

        if values not in GLOBAL_DF.columns:
            return f"Column '{values}' not found."

        fig = px.pie(
            GLOBAL_DF,
            names=names,
            values=values,
            title=f"{values} distribution across {names}"
        )

        return {
            "type": "pie",
            "names": names,
            "values": values,
            "figure": fig
        }

    except Exception as e:
        return f"Error generating pie chart: {str(e)}"


@tool
def box_plot_tool(column: str):
    """
    Generate box plot for outlier analysis.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        if column not in GLOBAL_DF.columns:
            return f"Column '{column}' not found."

        if not pd.api.types.is_numeric_dtype(GLOBAL_DF[column]):
            return f"Column '{column}' is not numeric."

        fig = px.box(
            GLOBAL_DF,
            y=column,
            title=f"Box Plot of {column}"
        )

        return {
            "type": "box",
            "column": column,
            "figure": fig
        }

    except Exception as e:
        return f"Error generating box plot: {str(e)}"


@tool
def heatmap_tool(dummy: str = "heatmap"):
    """
    Generate correlation heatmap.
    """

    global GLOBAL_DF

    try:

        if GLOBAL_DF is None:
            return "No dataframe loaded."

        numeric_df = GLOBAL_DF.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            return "Not enough numeric columns for heatmap."

        corr = numeric_df.corr()

        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values
        )

        return {
            "type": "heatmap",
            "figure": fig
        }

    except Exception as e:
        return f"Error generating heatmap: {str(e)}"