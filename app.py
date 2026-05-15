import streamlit as st

from workflows.analytics_graph import analytics_graph

from ui.hero_page import render_hero_page
from ui.analytics_page import render_analytics
from ui.chat_page import render_chat_page

from agents.insights_agent import InsightsAgent


st.set_page_config(
    page_title="Autonomous Data Analytics",
    layout="wide"
)


if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = {}

if "insights_generated" not in st.session_state:
    st.session_state.insights_generated = False


# SIDEBAR NAVIGATION

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Analytics Dashboard",
        "Dataset Chat"
    ]
)


# HOME PAGE

if page == "Home":

    render_hero_page()

    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        if st.button("Run Autonomous Analytics Workflow"):

            initial_state = {
                "file": uploaded_file,
                "logs": []
            }

            with st.spinner("Executing multi-agent analytics workflow..."):

                result = analytics_graph.invoke(initial_state)

                st.session_state.workflow_state = result

            st.success("Workflow completed successfully.")


# ANALYTICS DASHBOARD

elif page == "Analytics Dashboard":

    if st.session_state.workflow_state:

        render_analytics(st.session_state.workflow_state)

        # HUMAN IN THE LOOP

        if (
            "visualizations" in st.session_state.workflow_state
            and not st.session_state.insights_generated
        ):

            st.warning(
                "Review visualizations before generating final business insights."
            )

            if st.button("Show Hidden Insights"):

                with st.spinner("Generating executive insights..."):

                    insights_agent = InsightsAgent()

                    insights_result = insights_agent.run(
                        st.session_state.workflow_state
                    )

                    st.session_state.workflow_state.update(
                        insights_result
                    )

                    st.session_state.insights_generated = True

                st.success("Advanced business insights generated.")

    else:

        st.info("Run workflow from Home page first.")


# CHAT PAGE

elif page == "Dataset Chat":

    if "cleaned_df" in st.session_state.workflow_state:

        render_chat_page(
            st.session_state.workflow_state["cleaned_df"]
        )

    else:

        st.info("Run analytics workflow first.")