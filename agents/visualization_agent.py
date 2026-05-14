from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from tools.visualization_tools import (
    histogram_tool,
    scatter_tool,
    line_chart_tool,
    bar_chart_tool,
    pie_chart_tool,
    box_plot_tool,
    heatmap_tool,
    set_dataframe
)

load_dotenv()


class VisualizationAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.tools = [
            histogram_tool,
            scatter_tool,
            line_chart_tool,
            bar_chart_tool,
            pie_chart_tool,
            box_plot_tool,
            heatmap_tool
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, state):

        df = state['cleaned_df']

        set_dataframe(df)

        schema = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        prompt = f"""
        You are an autonomous visualization agent.

        Dataset Schema:
        {schema}

        Decide which visualization tools should be called.

        Rules:
        - Use heatmap for multiple numeric columns
        - Use histogram for distributions
        - Use scatter plots for relationships
        - Use box plots for outliers
        - Use bar charts for category analysis
        - Use line charts for date/time trends
        - Use pie charts for categorical proportions

        Generate the BEST possible visual analytics.
        """

        result = self.agent.invoke(prompt)

        return {
            "visualization_reasoning": result
        }