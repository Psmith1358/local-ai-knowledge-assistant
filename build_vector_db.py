from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

PDF_PATH = "data/SOP_Implementation_Success.pdf"
VECTOR_DB_PATH = "vectorstore"

print("Loading PDF...")

all_documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        print(f"Loading {file}...")

        loader = PyPDFLoader(
            os.path.join("data", file)
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file

        all_documents.extend(docs)

documents = all_documents

print(f"Loaded {len(documents)} pages from the PDF.")

print("Splitting PDF into text chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building local FAISS vector database...")

vectorstore = FAISS.from_documents(chunks, embeddings)

print("Saving vector database...")

vectorstore.save_local(VECTOR_DB_PATH)

print("Done. Vector database created successfully.")