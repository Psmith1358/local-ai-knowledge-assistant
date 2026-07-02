from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import shutil
import subprocess
import os

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
        source_file = doc.metadata.get(
            "source_file",
            "Unknown Document"
        )

        page = doc.metadata.get(
            "page",
            0
        ) + 1

        sources.append(
            f"{source_file} (Page {page})"
        )
    source_text = ", ".join(set(sources))

    context = "\n\n".join([doc.page_content for doc in docs])

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
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"

    print("******** UPLOAD ENDPOINT HIT ********", flush=True)
    print(f"Uploading file: {file.filename}", flush=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("******** REBUILDING VECTOR DATABASE ********", flush=True)

    subprocess.run(
        ["python", "build_vector_db.py"],
        check=True
    )

    print("******** VECTOR DATABASE REBUILD COMPLETE ********", flush=True)

    return {
        "message": "Document uploaded successfully and knowledge base updated!",
        "filename": file.filename
    }
@app.get("/documents")
def list_documents():
    files = []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            files.append(file)

    return {
        "documents": files
    }
@app.get("/delete-document")
def delete_document(filename: str):

    file_path = os.path.join("data", filename)

    if os.path.exists(file_path):
        os.remove(file_path)

        subprocess.run(
            ["python", "build_vector_db.py"],
            check=True
        )

        return {
            "message":
            "Document deleted and knowledge base updated."
        }

    return {
        "message":
        "Document not found."
    }