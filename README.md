# AI-Powered Financial Document Analyst

A production-level RAG (Retrieval-Augmented Generation) application 
that allows users to upload financial PDFs and ask questions in plain 
English — powered by GPT-4, LangChain, and ChromaDB.

## 🎯 What It Does

- Upload any financial PDF (annual reports, 10-K filings, earnings transcripts)
- Ask questions in plain English
- Get accurate, cited answers powered by GPT-4
- View source passages that support each answer

## 🏗️ Architecture

User → Streamlit UI → FastAPI Backend → LangChain RAG Pipeline
→ ChromaDB (Vector Search)
→ GPT-4 (Answer Generation)

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| LLM | GPT-4 (OpenAI) |
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | OpenAI text-embedding-ada-002 |
| PDF Parsing | PyMuPDF |

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Shivachowdoju/financial-doc-analyst.git
cd financial-doc-analyst
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:

OPENAI_API_KEY=your-openai-api-key-here

5. Start FastAPI backend:
```bash
uvicorn app.main:app --reload --port 8000
```

6. Start Streamlit frontend:
```bash
streamlit run frontend/streamlit_app.py
```

7. Open browser at `http://localhost:8501`

## 📁 Project Structure

financial-doc-analyst/
├── app/
│   ├── ingest.py        # PDF processing + ChromaDB storage
│   ├── query.py         # RAG query pipeline
│   └── main.py          # FastAPI REST API
├── frontend/
│   └── streamlit_app.py # Streamlit web UI
├── data/uploads/        # Uploaded PDFs
├── vectorstore/         # ChromaDB embeddings
├── requirements.txt
└── README.md

## 💡 How RAG Works

1. **Ingestion** — PDF parsed → split into 500-char chunks → 
   converted to vectors → stored in ChromaDB
2. **Query** — Question converted to vector → top 3 similar 
   chunks retrieved → GPT-4 generates grounded answer

## 🔒 Security

- API keys stored in `.env` file (never committed to Git)
- `.gitignore` protects sensitive files

## 👨‍💻 Author

Built as part of a 100-day Gen AI Engineering journey.

