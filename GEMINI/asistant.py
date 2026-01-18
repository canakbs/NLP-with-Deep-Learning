import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Doğru endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

def get_gemini_response(prompt: str) -> str:
    # Doğru payload yapısı
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            response_json = response.json()
            return response_json['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            raise Exception(f"Error parsing response JSON: {e}\nResponse: {response.text}")
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


if __name__ == "__main__":
    prompt = input("Kullanıcı Sorusu: ")
    try:
        response_text = get_gemini_response(prompt)
        print("\nGemini Response:")
        print(response_text)
    except Exception as e:
        print(f"Error: {e}")