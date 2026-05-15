from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from tools.cleaning_tools import (
    set_dataframe,
    get_dataframe,
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

        self.prompt = PromptTemplate.from_template("""
        You are an autonomous data cleaning agent.

        You have access to the following tools:

        {tools}
        Use the following format:

        Question: the user query or task
        Thought: think carefully about what to do
        Action: one of [{tool_names}]
        Action Input: input to the selected tool
        Observation: result returned by the tool
        ... (this Thought/Action/Action Input/Observation can repeat multiple times)
        Thought: I now know the final answer
        Final Answer: provide a clear and concise final response

        Begin!

        Question: {input}

        Thought: {agent_scratchpad}
        """)

        react_agent = create_react_agent(
            self.llm,
            self.tools,
            self.prompt
        )  
        self.agent = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def run(self, state):

        df = state['uploaded_df']

        set_dataframe(df)

        query = f"""
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

        response = self.agent.invoke({
            "input": query
        })

        cleaned_df = get_dataframe()

        return {
        "cleaned_df": cleaned_df,
        "cleaning_reasoning": response
    }