from langchain.agents import AgentType
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
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


class VisualizationAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
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

        prompt = """
            You are an autonomous visualization agent.

            You have access to visualization tools:

            {tools}

            Use the following format:

            Question: user task
            Thought: reasoning
            Action: tool selection
            Action Input: tool input
            Observation: result
            ...
            Thought: final reasoning
            Final Answer: final answer

            Question: {input}
            Thought:{agent_scratchpad}
            """
        

        react_agent = create_react_agent(
            self.llm,
            self.tools,
            self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

    def run(self, state):

        df = state['cleaned_df']

        set_dataframe(df)

        schema = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        response = self.agent_executor.invoke({
            "input": f"""
            Analyze dataset schema and autonomously decide:

            - which visualizations should be generated
            - which chart types are most suitable
            - which relationships should be analyzed

            Dataset Schema:
            {schema}

            Responsibilities:

            - generate distributions
            - generate correlation analysis
            - generate trend charts
            - generate outlier visualizations
            - generate categorical analysis charts

            Use visualization tools autonomously.
            """
        })

        return {
            "visualization_reasoning": response
        }