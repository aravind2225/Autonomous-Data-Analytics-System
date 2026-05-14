from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from tools.eda_tools import (
    set_dataframe,
    descriptive_statistics_tool,
    missing_values_tool,
    correlation_analysis_tool,
    categorical_summary_tool
)

load_dotenv()


class EDAAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.agent = initialize_agent(
            tools=[
                descriptive_statistics_tool,
                missing_values_tool,
                correlation_analysis_tool,
                categorical_summary_tool
            ],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, state):

        df = state['cleaned_df']

        set_dataframe(df)

        prompt = f"""
        You are an autonomous exploratory data analysis agent.

        Analyze the dataframe and decide:
        - which statistical summaries are needed
        - whether correlation analysis is required
        - whether categorical summaries should be generated

        Use tools intelligently.
        """

        result = self.agent.invoke(prompt)

        return {
            "eda_results": result
        }