import streamlit as st
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent 
from openai import OpenAI
from langchain.schema import SystemMessage

st.title('Hermes 2 Pro on MLX - Tool Use')

# Point to the local server
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

system_message = "<|im_start|>system You are a function calling AI model named Hermes 2 Pro. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {'type': 'function', 'function': {'name': 'multiply', 'description': 'Multiply two integers together.', 'parameters': {'type': 'object', 'properties': {'first_int': {'type': 'integer'}, 'second_int': {'type': 'integer'}}, 'required': ['first_int', 'second_int']}}} {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.', 'parameters': {'type': 'object', 'properties': {'first_int': {'type': 'integer'}, 'second_int': {'type': 'integer'}}, 'required': ['first_int', 'second_int']}}} {'type': 'function', 'function': {'name': 'exponentiate', 'description': 'Exponentiate the base to the exponent power.', 'parameters': {'type': 'object', 'properties': {'base': {'type': 'integer'}, 'exponent': {'type': 'integer'}}, 'required': ['base', 'exponent']}}} </tools> Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows: <tool_call> {'arguments': <args-dict>, 'name': <function-name>}</tool_call> <|im_end|>"

# tools = [multiply, add, exponentiate]

# # Construct the OpenAI Tools agent
# agent = create_openai_tools_agent(llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            try:
                stream = llm.chat.completions.create(
                    model='hermes-2-pro',
                    messages=[
                        {"system": system_message, "role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")