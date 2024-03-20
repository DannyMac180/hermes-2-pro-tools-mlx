import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import initialize_agent, tool
from langchain.llms import OpenAI
from langchain_core.messages import SystemMessage
from exa_py import Exa
load_dotenv()

st.title('Hermes 2 Pro on MLX - Tool Use')

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

exa = Exa(os.environ['EXA_API_KEY'])

@tool
def search(query: str):
    """Search for a webpage based on the query."""
    return exa.search(f"{query}", use_autoprompt=True, num_results=5)

tools = [search]
agent = initialize_agent(tools, client, agent='zero-shot-react-description', verbose=True, handle_parsing_errors=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    system_message = SystemMessage(
      content="You are a an AI assistant built on top of the Hermes 2 Pro LLM built by Nous Research. You have the ability to use the 'search' tool to search the web when appropriate."
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.markdown(response)
    st.session_state.messages.append({ "system": system_message, "role": "assistant", "content": response})

