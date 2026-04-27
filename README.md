# HaleAI - RAG-based Medical Chatbot

HaleAI is a medical RAG chatbot that uses a Python backend, Google Gemini, Pinecone, LangChain, and a static HTML frontend.

## What is used

- Python backend with FastAPI and Uvicorn
- Google Gemini for answer generation
- Pinecone for vector storage and retrieval
- LangChain, PyPDF, and Sentence Transformers for document processing
- A static frontend in [frontend/index.html](frontend/index.html) that loads React, Tailwind, Babel, and Recharts from CDNs

## Repository layout

```text
backend/
  backend_api.py      # FastAPI server used by the frontend
  main.py             # CLI entry point for one-time knowledge-base setup
  chatbot.py          # RAG orchestration
  data_processor.py   # PDF loading, chunking, embeddings
  vector_store.py     # Pinecone integration
  llm_handler.py      # Gemini / fallback generation
  config.py           # Environment and runtime configuration
frontend/
  index.html          # Static browser app
data/
  *.pdf               # Your source documents
```

## Prerequisites

- Python 3.8 or newer
- A Google API key that can access Gemini models
- A Pinecone API key
- Internet access for model and CDN requests

## Environment variables

Create a root `.env` file with your keys:

```env
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
```

Notes:
- `.env` is already ignored by git.
- `GEMINI_MODEL` is optional, but it is the cleanest way to switch models.
- The code defaults to a supported Gemini model if you do not set `GEMINI_MODEL` explicitly.

## Install dependencies

```bash
cd /workspaces/RAG-based-Medical-Chatbot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

What this does:
- Creates an isolated Python environment in `.venv`
- Installs the backend packages needed by the API, document processing, Pinecone, and Gemini integration

## End-to-end run flow

### 1. Add your PDFs

Put your medical PDF files in the [data](data) folder.

### 2. Build the knowledge base once

Run the CLI setup flow if you are adding documents for the first time or updating them:

```bash
cd /workspaces/RAG-based-Medical-Chatbot
. .venv/bin/activate
python backend/main.py
```

Then choose:
- `1` to set up the knowledge base

What this does:
- Loads PDFs from `data/`
- Splits them into chunks
- Creates embeddings
- Uploads them to Pinecone

Why this is necessary:
- The FastAPI server connects to the existing Pinecone index. It does not ingest PDFs on every startup.

### 3. Start the backend API

```bash
cd /workspaces/RAG-based-Medical-Chatbot
. .venv/bin/activate
python backend/backend_api.py
```

What this does:
- Starts the FastAPI server on `http://localhost:8000`
- Exposes chat, streaming chat, health, analytics, and conversation endpoints

Why this is necessary:
- The frontend sends requests to this API.

### 4. Open the frontend

Serve [frontend/index.html](frontend/index.html) with a local web server, for example:

```bash
cd /workspaces/RAG-based-Medical-Chatbot/frontend
python3 -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

What this does:
- Serves the static HTML app from a local HTTP origin

Why this is necessary:
- The frontend uses `fetch()` to call the backend, and browsers handle that more reliably from an HTTP server than from a raw file URL.

## Running in Codespaces

In GitHub Codespaces:

1. Create `.venv` and install backend dependencies.
2. Add your `.env` file in the repository root.
3. Run the one-time knowledge-base setup from `backend/main.py` if your Pinecone index is empty.
4. Start `backend/backend_api.py`.
5. Serve `frontend/index.html` with `python3 -m http.server 5500` or Live Server.

## Troubleshooting

- If the backend exits immediately, check that `.env` contains `PINECONE_API_KEY` and `GEMINI_API_KEY`.
- If you see import errors, make sure you installed the packages into `.venv`.
- If the frontend shows connection errors, confirm the backend is running on port `8000`.
- If the knowledge base is empty, rerun the setup step from `backend/main.py` after adding PDFs.

## Technologies used

- FastAPI
- Uvicorn
- Google Generative AI
- Pinecone
- LangChain ecosystem
- Sentence Transformers
- PyPDF
- Static HTML, React CDN, Tailwind CDN, Babel standalone, Recharts CDN

## Disclaimer

This project is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
