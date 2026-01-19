from langchain.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def get_stock_info(ticker: str) -> str:
    """
    Hisse senedi fiyat bilgilerini getirir.
    
    Args:
        ticker: Hisse senedi sembolü (örn: AAPL, GOOGL)
    
    Returns:
        Hisse senedi fiyat bilgileri
    """
    try:
        api_key = os.getenv("FINHUB_API_KEY")

        if not api_key:
            return "API key bulunamadı. .env dosyasını kontrol edin."
        
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
        
        response = requests.get(url)

        if response.status_code != 200:
            return f"HATA: API isteği başarısız - Status Code: {response.status_code}"
        
        data = response.json()

        # Güncel: c, open: o, high: h, low: l, previous close: pc
        current = data.get("c")
        open_ = data.get("o")
        high = data.get("h")
        low = data.get("l")
        prev_close = data.get("pc")

        # Veri kontrolü
        if current == 0 or current is None:
            return f"HATA: '{ticker}' için veri bulunamadı. Sembolü kontrol edin."

        # Değişim hesaplama
        change = current - prev_close if prev_close else 0
        change_percent = (change / prev_close * 100) if prev_close else 0

        return (f"{ticker} Hisse Bilgisi:\n"
                f"Güncel Fiyat: ${current:.2f}\n"
                f"Açılış: ${open_:.2f}\n"
                f"Gün İçi En Yüksek: ${high:.2f}\n"
                f"Gün İçi En Düşük: ${low:.2f}\n"
                f"Önceki Kapanış: ${prev_close:.2f}\n"
                f"Değişim: ${change:+.2f} ({change_percent:+.2f}%)"
        )

    except Exception as e:
        return f"HATA ALINDI: {str(e)}"


if __name__ == "__main__":
    # Tool olarak test etmek için
    print(get_stock_info.func("GOOGL"))
    
    # Veya invoke ile (dictionary formatında)
    # print(get_stock_info.invoke({"ticker": "GOOGL"}))