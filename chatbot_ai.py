import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-3.1-flash-lite")

print("Chatbot Gemini aktif.")
print("Ketik 'exit' untuk keluar.")

while True:
    user = input("Kamu: ")

    if user.lower() in ["exit", "keluar", "bye"]:
        print("Bot: Sampai jumpa.")
        break

    try:
        response = model.generate_content(user)
        print("Bot:", response.text)

    except Exception as e:
        print("Bot error:", e)