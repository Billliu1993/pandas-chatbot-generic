import pandas as pd
import json
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from pandas_chatbot_generic.agent import create_agent
from pandas_chatbot_generic.utils import load_csv_data, get_summary, load_json_data
from pandas_chatbot_generic.tools import plot_chart, get_calculator_tool


load_dotenv()
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0, verbose=True)
HISTORY_WINDOW = 10

st.set_page_config(page_title="Pandas Agent")
st.subheader("🤖 LangChain Pandas Agent Chatbot")
st.write("Upload a CSV file and query answers from your data.")
st_callback = StreamlitCallbackHandler(st.container())

# Initialize session state if not already done
if 'write_history' not in st.session_state:
    st.session_state.write_history = [{"role": "assistant", "text": "Hello! How can I assist you today?"}]
    st.session_state.chat_history = []

if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Function to handle button click
def clicked(button):
    st.session_state.clicked[button] = True

# Function to reset session state
def reset_session():
    st.session_state.write_history = [{"role": "assistant", "text": "Hello! How can I assist you today?"}]
    st.session_state.chat_history = []
    st.session_state.clicked = {1: False}
    st.session_state.uploaded_file = None

def convert_chat_history(chat_history):
    converted_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            converted_history.append({"role": "user", "text": message.content})
        elif isinstance(message, AIMessage):
            converted_history.append({"role": "assistant", "text": message.content})
    return converted_history

def convert_history_to_json(chat_history):
    return json.dumps(chat_history, indent=4)
    
# Create columns for horizontal layout
col1, col2, col3 = st.columns([1, 1, 1])

# Place buttons in columns
with col1:
    st.button("Get started", on_click=clicked, args=[1])

with col2:
    st.button("Reset query", on_click=reset_session)

if st.session_state.clicked[1]:
        st.session_state.uploaded_file = "file_uploaded"
        user_upload_csv = st.file_uploader("upload your data file in csv here", type=["csv"])
        user_upload_json = st.file_uploader("upload your schema file in json here", type=["json"])
        
        if (user_upload_csv is not None) and (user_upload_json is not None):
            
            data_df = load_csv_data(user_upload_csv)
            data_schema = load_json_data(user_upload_json)

            if data_schema != {}:
                selected_cols = list(data_schema.keys())
                data_df = data_df[selected_cols]
            
            info_df = get_summary(data_df)
            

            with st.expander("🔎 Dataframe Preview"):
                st.write(data_df.head(3))
            with st.expander("🔎 Dataframe Summary"):
                st.write(info_df)
            with st.expander("🔎 Schema Info"):
                st.json(data_schema)

            # archived for now due to answer quality
            # math_tool = get_calculator_tool(llm)

            agent_executor = create_agent(
                llm=llm, df=data_df, schema=data_schema, extra_tools=[plot_chart])
            
            if len(st.session_state.chat_history) > 0:
                for message in convert_chat_history(st.session_state.chat_history): 
                    with st.chat_message(message["role"]): 
                        st.markdown(message["text"]) 
            # test 
            query = st.chat_input("Enter a query:") 
            # Execute Button Logic
            if query:
                with st.chat_message("user"): 
                    st.markdown(query) 

                try:
                    with st.chat_message("assistant"): 
                        st_callback = StreamlitCallbackHandler(st.container())
                        response = agent_executor.invoke(
                            {"input": query, "chat_history": st.session_state.chat_history[-HISTORY_WINDOW:]},
                            {"callbacks": [st_callback]}
                            )
                        st.markdown(response["output"])

                    st.session_state.chat_history.extend(
                        [
                            HumanMessage(content=query),
                            AIMessage(content=response["output"])
                        ],
                    )

                except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

with col3:
    st.download_button(
        label="Download chat",
        data=convert_history_to_json(convert_chat_history(st.session_state.chat_history)),
        file_name='chat_history_%s.json' % datetime.now().strftime("%Y%m%d_%H%M%S"),
        mime='application/json'
    )
    
# streamlit run streamlit_app.py
