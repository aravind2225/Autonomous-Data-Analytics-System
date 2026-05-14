from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from tools.cleaning_tools import (
    set_dataframe,
    missing_value_tool,
    duplicate_removal_tool,
    outlier_detection_tool
)

load_dotenv()


class CleaningAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.agent = initialize_agent(
            tools=[
                missing_value_tool,
                duplicate_removal_tool,
                outlier_detection_tool
            ],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, state):

        df = state['uploaded_df']

        set_dataframe(df)

        prompt = f"""
        You are a professional data cleaning agent.

        Analyze dataset schema and autonomously decide:
        - whether duplicates exist
        - how to handle missing values
        - whether outlier detection is needed

        Dataset shape:
        {df.shape}

        Columns:
        {list(df.columns)}

        Use tools autonomously.
        """

        response = self.agent.invoke(prompt)

        return {
            "cleaned_df": df,
            "cleaning_reasoning": response
        }