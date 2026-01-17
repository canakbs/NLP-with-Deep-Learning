import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="Terminal Tourist Bot",
    page_icon="🗺️",
    layout="centered"
)

# --------------------
# MODEL
# --------------------
@st.cache_resource
def get_model():
    return ChatOllama(model="llama3.2:latest")

model = get_model()

# --------------------
# PROMPT
# --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful terminal tourist guide."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

# Runnable chain
chain = prompt | model

# --------------------
# MEMORY STORE
# --------------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# Memory wrapper
chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --------------------
# SESSION STATE INIT
# --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# --------------------
# UI
# --------------------
st.title("🗺️ Terminal Tourist Bot")

# Sidebar - Session ID girişi
with st.sidebar:
    st.header("User Settings")
    
    session_name = st.text_input(
        "Enter your name:",
        value=st.session_state.session_id or "",
        placeholder="e.g., Ali"
    )
    
    if st.button("Start New Session") or (session_name and not st.session_state.session_id):
        if session_name:
            st.session_state.session_id = session_name
            st.session_state.messages = []
            st.success(f"Welcome, {session_name}! 👋")
            st.rerun()
    
    if st.session_state.session_id:
        st.info(f"Current session: **{st.session_state.session_id}**")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.session_id in st.session_state.store:
                st.session_state.store[st.session_state.session_id].clear()
            st.rerun()
    
    st.divider()
    st.caption("Powered by Ollama & LangChain")

# Ana chat alanı
if not st.session_state.session_id:
    st.info("👈 Please enter your name in the sidebar to start chatting!")
else:
    # Chat mesajlarını göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me about tourist destinations..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Bot yanıtı - streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in chat.stream(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            ):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Bot mesajını kaydet
        st.session_state.messages.append({"role": "assistant", "content": full_response})