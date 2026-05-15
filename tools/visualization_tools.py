from langchain.tools import tool
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


GLOBAL_DF = None


def set_dataframe(df):
    global GLOBAL_DF
    GLOBAL_DF = df

@tool
def histogram_tool(column: str):
    """Generate histogram for numeric column."""

    fig = px.histogram(GLOBAL_DF, x=column)

    return fig


@tool
def scatter_tool(columns: str):
    """Generate scatter plot using two numeric columns separated by comma."""

    x, y = columns.split(",")

    fig = px.scatter(
        GLOBAL_DF,
        x=x.strip(),
        y=y.strip()
    )

    return fig


@tool
def line_chart_tool(columns: str):
    """Generate line chart for time-series analysis."""

    x, y = columns.split(",")

    fig = px.line(
        GLOBAL_DF,
        x=x.strip(),
        y=y.strip()
    )

    return fig


@tool
def bar_chart_tool(columns: str):
    """Generate bar chart using categorical and numeric columns."""

    x, y = columns.split(",")

    fig = px.bar(
        GLOBAL_DF,
        x=x.strip(),
        y=y.strip()
    )

    return fig


@tool
def pie_chart_tool(columns: str):
    """Generate pie chart for categorical proportions."""

    names, values = columns.split(",")

    fig = px.pie(
        GLOBAL_DF,
        names=names.strip(),
        values=values.strip()
    )

    return fig


@tool
def box_plot_tool(column: str):
    """Generate box plot for outlier analysis."""

    fig = px.box(GLOBAL_DF, y=column)

    return fig


@tool
def heatmap_tool(dummy: str = "heatmap"):
    """Generate correlation heatmap."""

    corr = GLOBAL_DF.select_dtypes(include='number').corr()

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values
    )

    return fig