import streamlit as st


def render_analytics(state):

    st.header("Analytics Workflow Dashboard")

    st.success("Workflow execution completed")

    if 'metadata' in state:

        with st.expander("Dataset Metadata", expanded=True):
            st.json(state['metadata'])

    if 'cleaning_reasoning' in state:

        with st.expander("Cleaning Agent Reasoning"):
            st.write(state['cleaning_reasoning'])

    if 'eda_results' in state:

        with st.expander("EDA Agent Results"):
            st.write(state['eda_results'])

    if 'statistics_reasoning_trace' in state:

        with st.expander("Statistical Agent Reasoning"):
            st.write(state['statistics_reasoning_trace'])

    if 'visualization_reasoning' in state:

        with st.expander("Visualization Agent Reasoning"):
            st.write(state['visualization_reasoning'])

    if 'insights' in state:

        st.subheader("Business Intelligence Insights")

        st.write(state['insights'])

    if 'visualizations' in state:

        st.subheader("Generated Visualizations")

        for name, fig in state['visualizations'].items():
            st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Approve Visualizations"):
            st.session_state.visualization_approved = True
            st.success("Visualizations approved")

    with col2:

        if st.button("Generate Final Insights"):
            st.session_state.insights_approved = True
            st.success("Insights generation approved")