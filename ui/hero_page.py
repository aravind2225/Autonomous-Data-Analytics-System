import streamlit as st


def render_hero_page():

    st.title("Autonomous Data Analytics Platform")

    st.markdown(
        """
        ### Enterprise Multi-Agent Analytics System

        Features:

        - Autonomous AI Agents
        - LangGraph Orchestration
        - Dynamic Tool Calling
        - ReAct Reasoning
        - Statistical Intelligence
        - Interactive Visualizations
        - Hybrid RAG Pipeline
        - Context-Aware Dataset Chat
        - Human Approval Workflow
        - Business Intelligence Insights
        """
    )

    st.info(
        "Upload a structured dataset to begin autonomous analytics."
    )