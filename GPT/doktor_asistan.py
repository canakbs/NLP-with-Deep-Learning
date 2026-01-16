"""
Gerekli kütüphaneler:
pip install langchain-openai langchain-core python-dotenv
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Ortam değişkenlerini yükle
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# LLM modelini başlat
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

# Konuşma geçmişini tutmak için liste
conversation_history = []

# Kullanıcı bilgilerini al
name = input("Adınız: ")
age = input("Yaşınız: ")

# Sistem mesajı oluştur
system_message = SystemMessage(content=(
    f"Sen bir doktor asistanısın. Hasta {name}, {age} yaşında bir kişi. "
    "Sağlık sorunları hakkında konuşmak istiyor. "
    "Yaşına uygun tavsiyeler ver ve ismiyle hitap et. "
    "Profesyonel ama samimi bir dil kullan."
))

# Sistem mesajını geçmişe ekle
conversation_history.append(system_message)

print(f"\n{'='*50}")
print("Doktor Asistanına Hoşgeldiniz!")
print(f"{'='*50}\n")
print("Sağlık sorunlarınızı paylaşabilirsiniz. Çıkmak için 'exit' yazabilirsiniz.\n")
# Ana döngü
while True:
    user_msg = input(f"{name}: ")
    
    if user_msg.lower() in ["exit", "quit", "çıkış", "cikis"]:
        print("\nGörüşmeniz sonlandırıldı. İyi günler dileriz!")
        break
    
    # Boş mesaj kontrolü
    if not user_msg.strip():
        print("Lütfen bir mesaj girin.\n")
        continue
    
    try:
        # Kullanıcı mesajını geçmişe ekle
        conversation_history.append(HumanMessage(content=user_msg))
        
        # LLM'den yanıt al
        response = llm.invoke(conversation_history)
        
        # AI yanıtını geçmişe ekle
        conversation_history.append(AIMessage(content=response.content))
        
        print(f"\nDoktor Asistanı: {response.content}\n")
        
    except Exception as e:
        print(f"\nBir hata oluştu: {str(e)}\n")
        print("Lütfen tekrar deneyin.\n")