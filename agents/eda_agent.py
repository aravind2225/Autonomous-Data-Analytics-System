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
        You are an autonomous exploratory data analysis agent.

        You have access to the following analytical tools:
        {tools}

        Use the following format:

        Question: the user task or query
        Thought: reason about the analysis approach
        Action: one of [{tool_names}]
        Action Input: input to the selected tool
        Observation: result from the tool
        ... (this process can repeat multiple times)
        Thought: I now know the final answer
        Final Answer: provide analytical insights clearly

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
            max_iterations=10
        )

    def run(self, state):

        df = state['cleaned_df']

        set_dataframe(df)

        response = self.agent_executor.invoke({
            "input": f"""
            Perform exploratory data analysis.

            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}

            Responsibilities:

            - generate descriptive statistics
            - analyze missing values
            - detect correlations
            - summarize categorical features
            - explain analytical findings

            Use tools autonomously.
            """
        })

        return {
            "eda_results": response
        }