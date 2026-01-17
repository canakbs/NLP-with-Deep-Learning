# to start
#uvicorn doctor_assistant_api:app --reload
#paste this to bash

import requests
API_URL = "http://127.0.0.1:8000/chat"

name=input("Adiniz: ")
age=input("Yasiniz: ")

print("Merhaba, sohbet başladı")


while True:

    user_msg=input(f"{name}: ")
    if user_msg.lower() in ['exit', 'quit']:
        print("Sohbet sonlandırıldı.")
        break

    payload = {
        "name": name,
        "age": age,
        "message": user_msg
    }

    try:
        res=requests.post(API_URL, json=payload,timeout=30)

        if res.status_code == 200:
            print(f"Doktor Asistan: {res.json().get('response') }")
        else:
            print("hata:",res.status_code,res.text)
    except requests.exceptions.RequestException as e:
        print("hata:", e)