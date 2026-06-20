from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

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

@app.get("/ask-document")
def ask_document(question: str):

    docs = vectorstore.similarity_search(question, k=3)

    sources = []

    for doc in docs:
        if "page" in doc.metadata:
            sources.append(f"Page {doc.metadata['page'] + 1}")

    context = "\n\n".join([doc.page_content for doc in docs])
    source_text = ", ".join(set(sources))

    prompt = f"""
You are a concise, professional S&OP knowledge assistant.

Answer the user's question using only the provided document context.
If the answer is not found in the context, say that the document does not provide enough information.

Keep the answer clear, useful, and under 200 words.

Document Context:
{context}

Question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return {
        "answer": result["response"],
        "sources": source_text
    }