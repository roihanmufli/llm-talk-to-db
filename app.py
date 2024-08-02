import streamlit as st
import os
from src.data_prep import Prep
from src.agent import SQLAgent
import tempfile
from pathlib import Path
import shutil
# extracting text from document

# @st.cache_resource(show_spinner=False)
def get_example_query():
    ingestion = Prep()
    vector_store = ingestion.get_query_examples()
    return vector_store

def get_conversationchain(query,selected_option,vector):
    if selected_option == "gemini-1.5-pro-latest":
        model = "models/gemini-1.5-pro-latest"
    else:
        model = "models/gemini-1.5-flash"

    qna = SQLAgent(model,vector)
    results = qna.generate_response(
        query=query
    )
    return results

# def clear_vector_db():
#     st.session_state.messages = [{"role": "assistant", "content": "upload some documents and ask me a question"}]
#     abs_path = os.path.dirname(os.path.abspath(__file__))
#     CurrentFolder = str(Path(abs_path).resolve())
#     path = os.path.join(CurrentFolder, "database")
#     shutil.rmtree(path)

# generating response from user queries and displaying them accordingly
# def handle_question(question):
#     response=st.session_state.conversation({'question': question})
#     st.session_state.chat_history=response["chat_history"]
#     for i,msg in enumerate(st.session_state.chat_history):
#         if i%2==0:
#             st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "ask me a question"}]
    
def main():
    st.set_page_config(page_title="Chat with multiple DOCUMENTs",page_icon="🤖")
    st.title("AIDSU - Chat to database credit card transaction")
    st.write("Welcome to the chat!")
    st.session_state.example_query_selector = get_example_query()


    if "example_query_selector" not in st.session_state:
        st.session_state.example_query_selector = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask the question to talk to database credit card transaction"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.sidebar:

        options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
        selected_option = st.selectbox("Select Gemini Model:", options, index= 0)

        
    # Main content area for displaying chat messages
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    # st.sidebar.button('Clear VectorDB', on_click=clear_vector_db)
    
    user_question = st.chat_input("Ask a question...")

    if user_question and st.session_state.example_query_selector:
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        response = get_conversationchain(user_question,selected_option,st.session_state["example_query_selector"])

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()