import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_openai_functions_agent
from langchain.agents.agent import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# TOOLS
from tools.search_tool import search
from tools.currency_converter import convert_usd_to_try
from tools.market_api import get_stock_info

load_dotenv()

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

tools = [search, convert_usd_to_try, get_stock_info]

prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen deneyimli bir yatırım danışmanısın.
Kullanıcının finans ve yatırım sorularına doğru araçları kullanarak cevap ver.

Kurallar:
1. Tavsiye verme, sadece bilgi ver
2. Gerekirse birden fazla araç kullan
3. Cevaplar Türkçe olsun
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("YATIRIM DANIŞMANI AI AGENT'İNA HOŞGELDİNİZ")
    print("="*60)
    print("Komutlar:")
    print("  - Hisse bilgisi için: 'Apple hissesi kaç dolar?'")
    print("  - Döviz çevirisi için: '100 doları TL'ye çevir'")
    print("  - Arama için: 'Tesla hakkında son haberler'")
    print("  - Çıkmak için: 'q' veya 'quit'")
    print("="*60 + "\n")

    while True:
        try:
            query = input("\n💼 Sorunuz: ").strip()

            if query.lower() in ["q", "quit", "exit", "çık"]:
                print("\n👋 Görüşmek üzere! İyi günler.")
                break

            if not query:
                print("⚠️  Lütfen bir soru girin.")
                continue

            # agent_executor kullan, agent değil!
            response = agent_executor.invoke({"input": query})
            
            print(f"\n✅ Yanıt:\n{response['output']}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
            
        except Exception as e:
            print(f"\n❌ HATA: {e}\n")
            print("Lütfen tekrar deneyin veya farklı bir soru sorun.\n")