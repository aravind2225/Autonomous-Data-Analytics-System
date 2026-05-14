import streamlit as st
from agents.chat_agent import ChatAgent


if 'chat_agent' not in st.session_state:
    st.session_state.chat_agent = ChatAgent()


chat_agent = st.session_state.chat_agent



def render_chat_page(df):

    st.header("Context-Aware Dataset Chat")

    st.markdown(
        """
        Ask analytical questions like:

        - Who has maximum sales?
        - Show revenue trends
        - Compare regions
        - Detect anomalies
        - Plot distribution of salaries
        - Which category performs best?
        - Summarize customer behavior
        """
    )

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:

        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    prompt = st.chat_input("Ask your dataset question...")

    if prompt:

        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing dataset..."):

                response = chat_agent.chat(df, prompt)

                answer = response['answer']

                st.markdown(answer)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer
                })

        with st.expander("Conversation Context"):

            for msg in response['chat_history']:
                st.write(msg)