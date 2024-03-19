import streamlit as st
from openai import OpenAI

st.title('Hermes 2 Pro on MLX - Tool Use')

# Point to the local server
client = OpenAI(base_url="http://localhost:5000", api_key="not-needed")

completion = client.chat.completions.create(
  model="local-model", # this field is currently unused
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ],
  temperature=0.7,
  )

# Print the response
print(completion.choices[0].message)

with st.chat_message('Hello!'):
    st.text('This is a chat message')
