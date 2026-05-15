import os
from dotenv import load_dotenv

from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent
)

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from memory.chat_memory import ChatMemoryManager

from tools.query_tools import (
    set_dataframe,
    top_records_tool,
    grouping_tool,
    dataframe_summary_tool
)

from tools.rag_tools import (
    initialize_vectorstore,
    semantic_search_tool
)

load_dotenv()


class ChatAgent:

    def __init__(self):

        self.memory_manager = ChatMemoryManager()

        self.memory = self.memory_manager.get_memory()

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        self.agent_executor = None

    def setup_agent(self, df):

        set_dataframe(df)

        initialize_vectorstore(df)

        tools = [
            top_records_tool,
            grouping_tool,
            dataframe_summary_tool,
            semantic_search_tool
        ]

        prompt = PromptTemplate.from_template("""
        You are an intelligent dataset analytics assistant.

        You have access to the following tools:

        {tools}

        IMPORTANT RULES:

        - Use tools only when necessary.
        - Do not repeatedly call the same tool.
        - Always provide concise analytical responses.
        - Use chat history for follow-up context.
        - Always end with Final Answer.
        - Never loop indefinitely.

        Use the following format:

        Question: user query
        Thought: reasoning about next step
        Action: one of [{tool_names}]
        Action Input: input for tool
        Observation: result from tool
        ... (repeat if necessary)
        Thought: I now know the final answer
        Final Answer: final analytical response

        Begin!

        Previous conversation history:
        {chat_history}

        Question: {input}

        Thought: {agent_scratchpad}
        """)

        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def chat(self, df, query):

        try:

            if self.agent_executor is None:

                self.setup_agent(df)

            # =====================================================
            # PANDAS AGENT
            # =====================================================

            pandas_agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                allow_dangerous_code=True
            )

            dataframe_response = pandas_agent.invoke(query)

            # =====================================================
            # FINAL REASONING
            # =====================================================

            final_prompt = f"""
            User Query:
            {query}

            Dataframe Agent Result:
            {dataframe_response}

            Use chat history context if relevant.

            If needed:
            - perform semantic retrieval
            - perform grouping
            - perform aggregation
            - explain trends
            - summarize analytical findings
            - answer context-aware follow-up questions

            Generate final analytical response.
            """

            response = self.agent_executor.invoke({
                "input": final_prompt
            })

            return {

                "answer": response["output"],

                "chat_history":
                    self.memory_manager.get_messages()
            }

        except Exception as e:

            return {

                "answer":
                    f"Chat Agent Error: {str(e)}",

                "chat_history":
                    self.memory_manager.get_messages()
            }