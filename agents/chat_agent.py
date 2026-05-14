# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os

# load_dotenv()


# class ChatAgent:

#     def __init__(self):

#         self.llm = ChatGroq(
#             model="llama3-70b-8192",
#             api_key=os.getenv("GROQ_API_KEY"),
#             temperature=0
#         )

#     def chat(self, df, query):

#         agent = create_pandas_dataframe_agent(
#             self.llm,
#             df,
#             verbose=True,
#             allow_dangerous_code=True
#         )

#         response = agent.invoke(query)

#         return response


from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from memory.chat_memory import ChatMemoryManager

from tools.query_tools import (
    set_dataframe,
    top_records_tool,
    grouping_tool,
    dataframe_summary_tool
)

from tools.visualization_tools import (
    histogram_tool,
    scatter_tool,
    line_chart_tool,
    bar_chart_tool,
    pie_chart_tool,
    box_plot_tool,
    heatmap_tool
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

    def build_dataframe_agent(self, df):

        set_dataframe(df)

        initialize_vectorstore(df)

        tools = [
            top_records_tool,
            grouping_tool,
            dataframe_summary_tool,
            semantic_search_tool,
            histogram_tool,
            scatter_tool,
            line_chart_tool,
            bar_chart_tool,
            pie_chart_tool,
            box_plot_tool,
            heatmap_tool
        ]

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=15
        )

        return agent

    def chat(self, df, query):

        pandas_agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        dataframe_response = pandas_agent.invoke(query)

        conversational_agent = self.build_dataframe_agent(df)

        final_prompt = f"""
        User Query:
        {query}

        Dataframe Agent Result:
        {dataframe_response}

        Use chat history context if relevant.

        If needed:
        - perform semantic retrieval
        - generate visualization
        - perform grouping
        - perform aggregation
        - explain trends
        - answer context-aware follow-up questions

        Generate final analytical response.
        """

        response = conversational_agent.invoke({
            "input": final_prompt
        })

        return {
            "answer": response['output'],
            "chat_history": self.memory_manager.get_messages()
        }