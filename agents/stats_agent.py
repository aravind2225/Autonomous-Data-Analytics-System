from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from tools.stats_tools import (
    set_dataframe,
    t_test_tool,
    correlation_test_tool,
    anomaly_detection_tool
)




class StatsAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.tools = [
            t_test_tool,
            correlation_test_tool,
            anomaly_detection_tool
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

        df = state['cleaned_df']

        set_dataframe(df)

        numeric_cols = list(
            df.select_dtypes(include='number').columns
        )

        schema = {
            "numeric_columns": numeric_cols,
            "shape": df.shape
        }

        prompt = f"""
        You are an autonomous statistical analysis agent.

        Dataset Information:
        {schema}

        Your responsibilities:

        1. Decide whether statistical testing is needed
        2. Detect strong correlations
        3. Detect anomalies/outliers
        4. Perform hypothesis testing where useful
        5. Analyze variance patterns
        6. Identify statistically significant relationships

        Tool Usage Rules:

        - Use t_test_tool when comparing two numeric distributions
        - Use correlation_test_tool for relationship analysis
        - Use anomaly_detection_tool for outlier detection

        Generate detailed statistical reasoning.

        IMPORTANT:
        - Decide tools autonomously
        - Use multiple tools if needed
        - Explain why each statistical test is selected
        - Return analytical conclusions
        """

        response = self.agent.invoke(prompt)

        return {
            "statistical_results": response,
            "statistics_reasoning_trace": response
        }