from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


class InsightsAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def run(self, state):

        df = state['cleaned_df']

        eda_results = state['eda_results']
        stats_results = state['statistical_results']

        prompt = f"""
        You are an expert autonomous data analyst.

        Generate advanced business insights.

        Perform:
        - anomaly detection reasoning
        - trend analysis
        - dominant feature analysis
        - hidden pattern detection
        - business risk detection
        - opportunity identification
        - correlation interpretation
        - skewness analysis
        - outlier reasoning
        - executive summary generation

        Dataset shape:
        {df.shape}

        EDA Results:
        {eda_results}

        Statistical Results:
        {stats_results}

        Return highly detailed analytical insights.
        """
        

        response = self.llm.invoke(prompt)

        return {
            "insights": response.content
        }