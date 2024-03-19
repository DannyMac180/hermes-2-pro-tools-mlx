from mlxserver import MLXServer
import streamlit as st

server = MLXServer(model="NousResearch/Hermes-2-Pro-Mistral-7B")

with st.chat_message("user"):
    st.write("Hello 👋")