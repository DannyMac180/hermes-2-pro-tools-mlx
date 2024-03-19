import streamlit as st
from langchain.llms import OpenAI

st.title('Hermes 2 Pro on MLX - Tool Use')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.chat_message('Hello!'):
    st.text('This is a chat message')
