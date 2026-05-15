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

    if 'statistical_results' in state:

        with st.expander("Statistical Agent Results"):
            st.write(state['statistical_results'])

    if 'visualization_reasoning' in state:

        with st.expander("Visualization Agent Reasoning"):
            st.write(state['visualization_reasoning'])

    # VISUALIZATIONS

    if 'visualizations' in state:

        st.subheader("Generated Visualizations")

        for chart_name, fig in state['visualizations'].items():

            st.markdown(f"### {chart_name}")

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # INSIGHTS

    if 'insights' in state:

        st.subheader("Executive Business Insights")

        st.write(state['insights'])