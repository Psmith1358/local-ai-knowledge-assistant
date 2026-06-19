from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def root():
    return {"message": "Local AI Knowledge Assistant is running"}

@app.get("/ask")
def ask_ai(question: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": f"""
You are a concise, professional AI assistant.

Answer the user's question clearly and briefly.
Use bullet points when helpful.
Limit your response to about 150 words unless the user asks for more detail.

Question:
{question}
""",
            "stream": False
        }
    )

    result = response.json()

    return {
        "answer": result["response"]
    }