import streamlit as st

from workflows.analytics_graph import analytics_graph

from ui.hero_page import render_hero_page
from ui.analytics_page import render_analytics
from ui.chat_page import render_chat_page


st.set_page_config(
    page_title="Autonomous Data Analytics",
    layout="wide"
)


if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = {}


render_hero_page()

uploaded_file = st.file_uploader(
    "Upload CSV or XLSX",
    type=['csv', 'xlsx']
)


if uploaded_file:
    if st.button("Run Analytics Workflow"):
        initial_state = {
            "file": uploaded_file,
            "logs": []
        }

        result = analytics_graph.invoke(initial_state)

        st.session_state.workflow_state = result


if st.session_state.workflow_state:
    render_analytics(st.session_state.workflow_state)

    if 'cleaned_df' in st.session_state.workflow_state:
        render_chat_page(
            st.session_state.workflow_state['cleaned_df']
        )