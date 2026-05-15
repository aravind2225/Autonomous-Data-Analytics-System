from langchain_groq import ChatGroq
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import os

from tools.visualization_tools import (
    set_dataframe,
    get_numeric_columns,
    get_categorical_columns
)


class VisualizationAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def run(self, state):

        try:

            df = state["cleaned_df"]

            set_dataframe(df)

            visualizations = {}

            numeric_cols = get_numeric_columns()
            categorical_cols = get_categorical_columns()

            # =====================================================
            # HISTOGRAM
            # =====================================================

            if len(numeric_cols) >= 1:

                hist_col = numeric_cols[0]

                hist_fig = px.histogram(
                    df,
                    x=hist_col,
                    title=f"Distribution of {hist_col}",
                    template="plotly_white"
                )

                visualizations[
                    f"{hist_col}_distribution"
                ] = hist_fig

            # =====================================================
            # SCATTER PLOT
            # =====================================================

            if len(numeric_cols) >= 2:

                scatter_fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                    template="plotly_white"
                )

                visualizations[
                    "scatter_relationship"
                ] = scatter_fig

            # =====================================================
            # BAR CHART
            # =====================================================

            if (
                len(categorical_cols) >= 1
                and len(numeric_cols) >= 1
            ):

                grouped_df = (
                    df.groupby(categorical_cols[0])[numeric_cols[0]]
                    .mean()
                    .reset_index()
                )

                bar_fig = px.bar(
                    grouped_df,
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=f"Average {numeric_cols[0]} by {categorical_cols[0]}",
                    template="plotly_white"
                )

                visualizations[
                    "categorical_bar_chart"
                ] = bar_fig

            # =====================================================
            # PIE CHART
            # =====================================================

            if (
                len(categorical_cols) >= 1
                and len(numeric_cols) >= 1
            ):

                pie_df = (
                    df.groupby(categorical_cols[0])[numeric_cols[0]]
                    .sum()
                    .reset_index()
                )

                pie_fig = px.pie(
                    pie_df,
                    names=categorical_cols[0],
                    values=numeric_cols[0],
                    title=f"{numeric_cols[0]} Contribution by {categorical_cols[0]}"
                )

                visualizations[
                    "pie_distribution"
                ] = pie_fig

            # =====================================================
            # BOX PLOT
            # =====================================================

            if len(numeric_cols) >= 1:

                box_fig = px.box(
                    df,
                    y=numeric_cols[0],
                    title=f"Outlier Analysis of {numeric_cols[0]}",
                    template="plotly_white"
                )

                visualizations[
                    "outlier_boxplot"
                ] = box_fig

            # =====================================================
            # HEATMAP
            # =====================================================

            if len(numeric_cols) >= 2:

                corr = df[numeric_cols].corr()

                heatmap_fig = ff.create_annotated_heatmap(
                    z=corr.values,
                    x=list(corr.columns),
                    y=list(corr.index),
                    annotation_text=np.round(corr.values, 2),
                    colorscale="Viridis"
                )

                heatmap_fig.update_layout(
                    title="Correlation Heatmap"
                )

                visualizations[
                    "correlation_heatmap"
                ] = heatmap_fig

            # =====================================================
            # LLM REASONING
            # =====================================================

            reasoning_prompt = f"""
            You are an expert data visualization analyst.

            Analyze the generated visualizations and provide:

            - trend analysis
            - anomaly interpretation
            - relationship analysis
            - distribution interpretation
            - business insights from charts

            Dataset Shape:
            {df.shape}

            Numeric Columns:
            {numeric_cols}

            Categorical Columns:
            {categorical_cols}

            Generated Charts:
            {list(visualizations.keys())}

            Provide concise but insightful reasoning.
            """

            reasoning_response = self.llm.invoke(
                reasoning_prompt
            )

            return {

                "visualizations": visualizations,

                "visualization_reasoning":
                    reasoning_response.content
            }

        except Exception as e:

            return {

                "visualization_reasoning":
                    f"Visualization Agent Error: {str(e)}"
            }