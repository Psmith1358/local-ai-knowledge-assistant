# Local AI Knowledge Assistant

A locally hosted AI assistant built with FastAPI, Ollama, and Llama 3.1.

This project allows users to ask questions through a web-based chat interface and receive responses from a locally running large language model without relying on cloud-based AI services.

---

## Features

- Local Llama 3.1 inference using Ollama
- FastAPI backend
- Web-based chat interface
- Real-time AI responses
- Enter-to-send functionality
- Clear chat functionality
- GitHub version control

---

## Technology Stack

- Python
- FastAPI
- Ollama
- Llama 3.1
- HTML
- CSS
- JavaScript
- Git
- GitHub

---

## Architecture

User Question

↓

Frontend (HTML/CSS/JavaScript)

↓

FastAPI Backend

↓

Ollama API

↓

Llama 3.1

↓

Response Returned to User

---

## Screenshots

### Project Structure

![Project Structure](screenshots/project-structure.png)

### FastAPI Backend

![FastAPI Backend](screenshots/fastapi-ollama-code.png)

### Running Application

![Running Application](screenshots/assistant-homepage.png)

### AI Response Example

![AI Response](screenshots/assistant-response.png)

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/Psmith1358/local-ai-knowledge-assistant.git
```

Install dependencies:

```bash
pip install fastapi uvicorn requests
```

Start Ollama:

```bash
ollama run llama3.1
```

Start FastAPI:

```bash
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/frontend
```

---

## Future Enhancements

- PDF document ingestion
- Vector database integration
- Retrieval-Augmented Generation (RAG)
- Multi-document search
- Source citation support
- User authentication

---

## Author

Payton Smith

Master of Science in Information Systems
Certificate in AI for Business