import os
from dotenv import load_dotenv
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Assuming these are defined correctly in your local files
from memory.chat_memory import ChatMemoryManager
from tools.query_tools import (
    set_dataframe, top_records_tool, grouping_tool, dataframe_summary_tool
)
from tools.visualization_tools import (
    histogram_tool, scatter_tool, line_chart_tool, 
    bar_chart_tool, pie_chart_tool, box_plot_tool, heatmap_tool
)
from tools.rag_tools import (
    initialize_vectorstore, semantic_search_tool
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
        """Initializes tools and the agent executor once."""
        set_dataframe(df)
        initialize_vectorstore(df)

        tools = [
            top_records_tool, grouping_tool, dataframe_summary_tool,
            semantic_search_tool, histogram_tool, scatter_tool,
            line_chart_tool, bar_chart_tool, pie_chart_tool,
            box_plot_tool, heatmap_tool
        ]

        # Get the standard ReAct prompt from LangChain Hub
        # You can customize this prompt to tell the agent how to use the tools
        prompt = PromptTemplate.from_template("""
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tools_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
        )

        # Create the ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        # Create the executor (this holds the memory and handles the loop)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

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