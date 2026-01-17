# ollama_model = llama3.2:latest

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --------------------
# MODEL
# --------------------
model = ChatOllama(model="llama3.2:latest")

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
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Memory wrapper
chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --------------------
# TERMINAL LOOP
# --------------------
name = input("Enter your name: ")
session_id = name  # basit kullanım için

print(f"\nWelcome to the Terminal Tourist Bot, {name}!")
print("Type 'exit' or 'quit' to leave the chat.\n")
print("Bot: Hello! I'm your terminal tourist guide. How can I assist you today?\n")
while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the terminal tourist bot. Goodbye!")
        break

    print("Bot: ", end="", flush=True)
    
    full_response = ""
    for chunk in chat.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ):
        content = chunk.content
        print(content, end="", flush=True)
        full_response += content
    
    print()  # Yeni satır için