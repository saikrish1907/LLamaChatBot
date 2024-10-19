import streamlit as st
from langchain.schema import(SystemMessage, HumanMessage, AIMessage)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

## Initalise the page using streamlit app
def init_page():
  st.set_page_config(page_title="ChatBot", page_icon="🤖")
  st.sidebar.image('logo.jpg', width=100)
  st.sidebar.title("AI ChatBot")
  st.sidebar.subheader("Intelligent bot to answer all your questions at the tap of your finger")

def init_messages():
  clear_button = st.sidebar.button("Clear Conversation", key="clear")
  if clear_button or "messages" not in st.session_state:
    #Setting the system message to make sure that BOT understand its role and behavior
    st.session_state.messages = [
      SystemMessage(
        content="You are intelligent AI assistant. Reply your answer in markdown format."
      )
    ]

def initialize_model():
  return LlamaCPP(
    model_path="llama-2-7b-chat.Q2_K.gguf",
    temperature=0.7,
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers":64},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
  )

def get_answer(llm, prompt):
    response = llm.complete(prompt)
    response_text = response.text
    return response_text


def main():
  init_page()
  init_messages()
  llm = initialize_model()
  
  if user_input := st.chat_input("Input your question!"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("ChatBot is typing ..."):
      answer = get_answer(llm, user_input)
    st.session_state.messages.append(AIMessage(content=answer))
    

  messages = st.session_state.get("messages", [])
  for message in messages:
    if isinstance(message, AIMessage):
      with st.chat_message("AI Assistant"):
        st.markdown(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("Human"):
        st.markdown(message.content)

if __name__ == "__main__":
  main()
