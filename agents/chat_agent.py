from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class ChatAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def chat(self, df, query):

        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        response = agent.invoke(query)

        return response