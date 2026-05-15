from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

from tools.eda_tools import (
    set_dataframe,
    descriptive_statistics_tool,
    missing_values_tool,
    correlation_analysis_tool,
    categorical_summary_tool
)


class EDAAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.tools = [
            descriptive_statistics_tool,
            missing_values_tool,
            correlation_analysis_tool,
            categorical_summary_tool
        ]

        self.prompt = PromptTemplate.from_template("""
        You are an exploratory data analysis agent.

        You have access to the following tools:

        {tools}

        IMPORTANT RULES:

        - Use only the necessary tools.
        - Do not repeatedly call the same tool.
        - After sufficient analysis, ALWAYS provide Final Answer.
        - Never loop indefinitely.
        - Maximum tool usage should usually be 3 to 4 actions.

        Use the following format:

        Question: the user task
        Thought: reasoning about next step
        Action: one of [{tool_names}]
        Action Input: tool input
        Observation: result from tool
        ... (repeat only if necessary)
        Thought: I now know the final answer
        Final Answer: provide concise analytical insights

        Begin!

        Question: {input}

        Thought: {agent_scratchpad}
        """)

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
            max_iterations=5
        )

    def run(self, state):

        try:

            df = state['cleaned_df']

            set_dataframe(df)

            query = f"""
            Perform exploratory data analysis on this dataset.

            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}

            Tasks:
            1. Generate descriptive statistics
            2. Analyze missing values
            3. Analyze correlations
            4. Summarize one important categorical column
            5. Provide final analytical insights

            Use only the necessary tools.
            Always end with Final Answer.
            """

            response = self.agent_executor.invoke({
                "input": query
            })

            return {
                "eda_results": response
            }

        except Exception as e:

            return {
                "eda_results": f"EDA Agent Error: {str(e)}"
            }