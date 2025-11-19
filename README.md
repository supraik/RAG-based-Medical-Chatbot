# HaleAI Medical Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for medical information powered by Google Gemini, Pinecone vector database, and a clean React frontend.

## 🚀 Quick Start

### Backend Setup

```bash
cd backend
python backend_api.py
```

Backend will start on http://localhost:8000

### Frontend Setup

```bash
cd frontend

# Option 1: Double-click start_server.bat (Windows)
# Option 2: PowerShell
./start_server.ps1

# Option 3: Manual
python -m http.server 5500
```

Frontend will open at http://localhost:5500/index.html

## 📋 Project Structure

```
HaleAI/
├── backend/
│   ├── backend_api.py      # FastAPI REST API
│   ├── chatbot.py          # Main chatbot logic
│   ├── llm_handler.py      # Gemini/HuggingFace LLM
│   ├── vector_store.py     # Pinecone vector DB
│   ├── rag_processor.py    # RAG pipeline & reranking
│   ├── data_processor.py   # PDF processing
│   ├── config.py           # Configuration
│   ├── requirements.txt    # Python dependencies
│   ├── src/
│   │   ├── helper.py       # Utility functions
│   │   └── prompt.py       # System prompts
│   └── data/               # Medical PDF documents
├── frontend/
│   ├── index.html          # React UI (self-contained)
│   ├── start_server.bat    # Windows launcher
│   ├── start_server.ps1    # PowerShell launcher
│   └── README.md           # Frontend documentation
└── README.md               # This file
```

## 🔧 Prerequisites

- **Python 3.8+** (3.11 recommended)
- **Pinecone Account** (free tier: https://app.pinecone.io/)
- **Google AI Studio API Key** (free: https://aistudio.google.com/app/apikey)
- **HuggingFace Token** (optional, for fallback: https://huggingface.co/settings/tokens)

## 📦 Installation

### 1. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n HaleAI python=3.11 -y
conda activate HaleAI

# OR using venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the `backend/` folder:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token  # Optional
```

### 4. Add Medical Documents

Place PDF files in `backend/data/` folder

### 5. Setup Vector Database (First Time Only)

```bash
cd backend
python setup.py
```

This will:

- Process your PDF documents
- Generate embeddings
- Upload to Pinecone vector store

## 🎯 Features

### Backend (FastAPI)

- ✅ **Google Gemini Integration** (gemini-1.5-flash)
- ✅ **HuggingFace Fallback** (google/flan-t5-small)
- ✅ **Pinecone Vector Store** (384-dim embeddings)
- ✅ **Cross-Encoder Reranking** (ms-marco-MiniLM-L-6-v2)
- ✅ **Streaming Responses** (Server-Sent Events)
- ✅ **Session Management** (conversation history)
- ✅ **CORS Enabled** (all frontend ports)

### Frontend (HTML/React)

- ✅ **No Build Required** (CDN-based React)
- ✅ **Streaming Chat UI** (real-time token display)
- ✅ **Markdown Rendering** (bold, italic, code, lists)
- ✅ **Conversation History** (maintained across messages)
- ✅ **Analytics Dashboard** (accuracy, latency metrics)
- ✅ **Dark Mode** (toggle)
- ✅ **Export Conversations** (JSON/TXT)
- ✅ **Responsive Design** (mobile-friendly)

## 🔌 API Endpoints

| Endpoint                         | Method | Description          |
| -------------------------------- | ------ | -------------------- |
| `/api/health`                    | GET    | Health check         |
| `/api/chat`                      | POST   | Non-streaming chat   |
| `/api/chat/stream`               | POST   | Streaming chat (SSE) |
| `/api/analytics`                 | GET    | System analytics     |
| `/api/conversations/{id}/export` | GET    | Export conversation  |

## ⚙️ Configuration

Edit `backend/config.py` to customize:

```python
# LLM Configuration
GEMINI_MODEL = "gemini-1.5-flash"           # Gemini model
USE_HF_FALLBACK = True                       # Enable HF fallback
HF_FALLBACK_MODEL = "google/flan-t5-small"   # Fallback model

# Vector Store
PINECONE_INDEX_NAME = "medical-chatbot"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
RETRIEVER_K = 8          # Documents to retrieve
RERANK_TOP_K = 3         # Documents after reranking
```

## 🧪 Testing

```bash
cd backend

# Test vector store connection
python test_pinecone.py

# Test generic RAG pipeline
python test_generic_rag.py

# Test environment variables
python test/test_env.py
```

## 📖 Usage

### Starting the Application

**Terminal 1 - Backend:**

```bash
cd backend
python backend_api.py
```

**Terminal 2 - Frontend:**

```bash
cd frontend
python -m http.server 5500
```

**Open Browser:**
http://localhost:5500/index.html

### Using the Chat

1. Type your medical question
2. Press Enter or click Send
3. Watch the response stream in real-time
4. Conversation context is maintained automatically

### Example Questions

- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What causes migraine headaches?"

## 🐛 Troubleshooting

### Backend Issues

**"Pinecone connection failed"**

- Check `PINECONE_API_KEY` in `.env`
- Verify index name matches `config.py`

**"Gemini API error"**

- Check `GOOGLE_API_KEY` in `.env`
- Verify API quota (free tier limits)
- Enable HuggingFace fallback

**"ModuleNotFoundError"**

- Activate virtual environment
- Run `pip install -r requirements.txt`

### Frontend Issues

**"Failed to fetch"**

- Ensure backend is running on port 8000
- Check CORS configuration in `backend_api.py`

**Streaming not working**

- Check browser console for errors
- Verify `/api/chat/stream` endpoint accessibility

**Markdown not rendering**

- Clear browser cache
- Refresh page (Ctrl+F5)

## 💡 Tips for Best Results

1. **First Query is Slow**: Model loading takes 20-60 seconds initially
2. **Be Specific**: Ask clear, focused medical questions
3. **Context Matters**: The chatbot maintains conversation history
4. **Check Sources**: Review the source documents provided with answers
5. **Rate Limits**: Free tier APIs have usage limits

## 🚢 Deployment

### Backend (FastAPI)

```bash
# Production server
pip install uvicorn[standard]
uvicorn backend_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

Simply host `frontend/index.html` on any static file server:

- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront

## 📊 System Requirements

**Minimum:**

- CPU: 2 cores
- RAM: 4 GB
- Disk: 2 GB free

**Recommended:**

- CPU: 4+ cores
- RAM: 8 GB
- Disk: 5 GB free

## 🔐 Security Notes

- Never commit `.env` file to git
- Use environment variables for production
- Implement rate limiting for production deployment
- Add authentication for sensitive medical data

## 📝 License

This project is for educational purposes. Consult with medical professionals for actual medical advice.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review backend logs in `backend/logs/`
3. Check browser console for frontend errors

## 🎓 Learn More

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Google Gemini API](https://ai.google.dev/docs)
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

**Built with ❤️ using FastAPI, React, Pinecone, and Google Gemini**
