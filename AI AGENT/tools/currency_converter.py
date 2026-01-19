from langchain.tools import tool
import requests

@tool
def convert_usd_to_try(amount: float) -> str:
    """
    USD'yi TRY'ye çevirir.
    
    Args:
        amount: Çevrilecek USD miktarı
    """
    try:
        if isinstance(amount, str):
            amount = float("".join(filter(lambda c: c.isdigit() or c == ".", amount)))
        
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"API HATASI {response.status_code}"

        data = response.json()
        rate = data["rates"]["TRY"]
        result = amount * rate
        
        return f"{amount} USD = {result:.2f} TRY (Kur: {rate:.2f})"
    
    except Exception as e:
        return f"HATA OLUSTU: {e}"

if __name__ == "__main__":
    test_amount = 100
    print(f"test amount: {test_amount}")
    # Dictionary formatında gönder
    print(convert_usd_to_try.invoke({"amount": test_amount}))