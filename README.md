# HaleAI - RAG-based Medical Chatbot 

A Retrieval-Augmented Generation (RAG) chatbot designed for medical document Q&A. Upload your medical PDFs and chat with them using state-of-the-art AI models.

## Features

- **PDF Processing**: Automatically extract and process medical documents
- **Vector Search**: Semantic search using Pinecone vector database
- **AI-Powered Responses**: Get accurate answers using Hugging Face's open-source LLMs
- **100% Free**: Uses free tier APIs (Pinecone + Hugging Face)
- **Easy Setup**: Simple CLI interface with step-by-step guidance

##  Prerequisites

- Python 3.8 or higher
- Internet connection (for API calls)
- Pinecone account (free tier available)
- Hugging Face account (free)

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/supraik/RAG-based-Medical-Chatbot.git
cd RAG-based-Medical-Chatbot
```

### 2. Set Up Virtual Environment

**Using venv:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

**Using conda (alternative):**
```bash
conda create -n HaleAI python=3.11 -y
conda activate HaleAI
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install packages individually:**
```bash
pip install -U "langchain>=0.3.7" "langchain-core>=0.3.7" "langchain-pinecone>=0.1.0" "langchain-openai>=0.2.0" packaging>=24.2
```

### 4. Get Your API Keys

#### Pinecone API Key
1. Go to [https://app.pinecone.io/](https://app.pinecone.io/)
2. Sign up for a free account
3. Create a new project
4. Copy your API key from the dashboard

#### Hugging Face Token
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "read" permissions
3. Copy the token

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_actual_pinecone_key
HF_TOKEN=your_actual_huggingface_token
```

### 6. Add Your Medical PDFs

Place your PDF files in the `data/` folder

##  Project Structure

```
haleai/
├── main.py                 # Main entry point
├── chatbot.py             # Chatbot logic
├── llm_handler.py         # LLM interaction handler
├── vector_store.py        # Pinecone vector store management
├── data_processor.py      # PDF processing and chunking
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
└── data/                  # Your medical PDF files
    └── your_medical_pdfs.pdf
```

## Usage

### Step 1: Setup (First Time Only)

```bash
python main.py
# Select option 1 (Setup)
```

This will:
- Process your PDF documents
- Create embeddings
- Upload to Pinecone vector database

**Note:** This step may take a few minutes depending on the size of your documents.

### Step 2: Chat with Your Documents

```bash
python main.py
# Select option 2 (Chat)
```

Now you can ask questions about your medical documents!

**Example Questions:**
- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What are the side effects of this medication?"

##  Configuration

You can customize the chatbot by editing `config.py`:

### Change the LLM Model

Default model: `mistralai/Mistral-7B-Instruct-v0.1`

**Alternative models:**

```python
# Fast and efficient
LLM_MODEL = "microsoft/phi-2"

# Good balance of speed and quality
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# High quality (requires HF license acceptance)
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
```

### Adjust Chunk Size

If you experience memory issues:

```python
CHUNK_SIZE = 500  # Reduce from default 1000
```

##  Troubleshooting

### First Query is Very Slow (20-60 seconds)

**This is normal!** The model needs to load for the first time. Subsequent queries will be much faster (3-10 seconds).

### Rate Limit Errors

- Free Hugging Face API allows ~100 requests/hour
- Wait a minute between queries if you hit limits
- Consider upgrading to Pro ($9/month) for unlimited access

### Model Not Working

Try these steps:
1. Check your API keys are correct in `.env`
2. Ensure you have internet connection
3. Try an alternative model (like `microsoft/phi-2`)
4. Check [Hugging Face status page](https://status.huggingface.co/)

### Embedding Issues

If embeddings cause problems:
- Reduce `CHUNK_SIZE` in `config.py`
- Make sure your PDFs are text-based (not scanned images)

##  Cost Breakdown

| Resource | Free Tier | Paid Option |
|----------|-----------|-------------|
| **Pinecone** | 1 index, 100K vectors | Starts at $70/month |
| **Hugging Face API** | ~100 requests/hour | Pro: $9/month (unlimited) |
| **Embeddings** | Unlimited (local) | Free |
| **Total Cost** | **$0** | Optional upgrades |

## 🔧 Technical Details

### Technologies Used

- **LangChain**: Framework for LLM applications
- **Pinecone**: Vector database for semantic search
- **Hugging Face**: Open-source LLM models
- **PyPDF**: PDF text extraction
- **Sentence Transformers**: Text embeddings

### How It Works

1. **Document Processing**: PDFs are split into manageable chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in Pinecone
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Response Generation**: Relevant context is sent to the LLM to generate answers

##  Best Practices

- **Ask Specific Questions**: Clear, focused questions get better answers
- **Wait Patiently**: First query takes longer (model loading)
- **Check Sources**: Always verify medical information from authoritative sources
- **Update Documents**: Re-run setup when adding new PDFs

##  Disclaimer

This chatbot is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding medical conditions.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

