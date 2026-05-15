from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
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

        self.prompt = PromptTemplate.from_template("""
You are an autonomous statistical analysis agent.

You have access to the following tools:

{tools}

Use the following format:

Question: the user query
Thought: think about the statistical approach
Action: one of [{tool_names}]
Action Input: tool input
Observation: tool result
... (repeat as necessary)
Thought: I now know the final answer
Final Answer: provide the final statistical interpretation

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

        try:

            df = state['cleaned_df']

            set_dataframe(df)

            numeric_cols = list(
                df.select_dtypes(include='number').columns
            )

            query = f"""
Perform statistical analysis on the dataset.

Dataset Shape:
{df.shape}

Numeric Columns:
{numeric_cols}

Responsibilities:

1. Decide whether statistical testing is needed
2. Detect strong correlations
3. Detect anomalies/outliers
4. Perform hypothesis testing where useful
5. Analyze variance patterns
6. Identify statistically significant relationships

Tool Usage Rules:

- Use t_test_tool when comparing numeric columns
- Use correlation_test_tool for relationship analysis
- Use anomaly_detection_tool for anomaly detection

Use tools autonomously wherever appropriate.
"""

            response = self.agent.invoke({
                "input": query
            })

            return {
                "statistical_results": response
            }

        except Exception as e:

            return {
                "statistical_results": f"Stats Agent Error: {str(e)}"
            }