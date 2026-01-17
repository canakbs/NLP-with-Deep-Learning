import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

# -------------------------
# APP INIT
# -------------------------
app = FastAPI()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY bulunamadı (.env dosyasını kontrol et)")

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=api_key
)

# -------------------------
# MEMORY STORE
# -------------------------
user_memories: Dict[str, ConversationBufferMemory] = {}

# -------------------------
# SCHEMAS
# -------------------------
class ChatRequest(BaseModel):
    name: str
    age: int
    message: str

class ChatResponse(BaseModel):
    response: str

# -------------------------
# ENDPOINT
# -------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    try:
        # Kullanıcıya özel hafıza oluştur
        if request.name not in user_memories:
            memory = ConversationBufferMemory(return_messages=True)

            system_prompt = (
                f"Sen bir doktor asistanısın. "
                f"Hasta adı: {request.name}. "
                f"Hasta yaşı: {request.age}. "
                f"Sağlık konularında sorular soruyor. "
                f"Yaşına uygun, güvenli, etik ve genel tıbbi tavsiyeler ver. "
                f"Kesin tanı koyma, doktora yönlendir. "
                f"Profesyonel ama samimi bir dil kullan ve ismiyle hitap et."
            )

            memory.chat_memory.messages.append(
                SystemMessage(content=system_prompt)
            )

            user_memories[request.name] = memory

        memory = user_memories[request.name]

        # Conversation Chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )

        # Model cevabı
        response_text = conversation.predict(input=request.message)

        # DEBUG (opsiyonel)
        print("\n--- MEMORY DUMP ---")
        for i, msg in enumerate(memory.chat_memory.messages, 1):
            print(f"{i}. {msg.type.upper()}: {msg.content}")

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
