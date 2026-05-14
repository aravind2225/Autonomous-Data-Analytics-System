from langchain.agents import AgentType
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
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
        self.tools = [
                missing_value_tool,
                duplicate_removal_tool,
                outlier_detection_tool
            ]

        prompt = """
        Answer the following questions as best you can.
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """

        react_agent = create_react_agent(
            self.llm,
            self.tools,
            prompt
        )  
        self.agent = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
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