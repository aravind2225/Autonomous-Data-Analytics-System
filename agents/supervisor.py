from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class SupervisorAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def decide_next_step(self, state):

        prompt = f"""
        You are a supervisor agent.

        Decide the next best analytics step.

        Current State:
        {list(state.keys())}

        Available Steps:
        - cleaning
        - eda
        - statistics
        - visualization
        - insights
        - chat

        Return only one next step.
        """

        response = self.llm.invoke(prompt)

        return response.content.strip().lower()