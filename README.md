# ResearchAgent — LangGraph Multi-Agent RAG System

A production-ready AI research agent built with **LangGraph**, **FAISS**, and **FastAPI**.
The agent intelligently routes queries between a local vector knowledge base and live web search,
grades retrieved content for relevance, and synthesizes grounded, cited answers.

**100% free to run** — uses Google Gemini (free tier) + DuckDuckGo search (no key needed).

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph Agent                │
│                                             │
│  ┌──────────┐    ┌────────────────────┐     │
│  │  Router  │───▶│  Retrieve (FAISS)  │     │
│  │  (LLM)   │    └────────────────────┘     │
│  │          │    ┌────────────────────┐     │
│  │          │───▶│   Web Search       │     │
│  └──────────┘    │   (Tavily API)     │     │
│                  └────────────────────┘     │
│                          │                  │
│                  ┌───────▼────────┐         │
│                  │  Relevance     │         │
│                  │  Grader (LLM)  │         │
│                  └───────┬────────┘         │
│                          │                  │
│                  ┌───────▼────────┐         │
│                  │  Synthesizer   │         │
│                  │  (LLM + RAG)   │         │
│                  └───────┬────────┘         │
└──────────────────────────┼──────────────────┘
                           │
                    Grounded Answer
                    + Citations
                    + Step Trace
```

### Agent Nodes

| Node | Role | Model |
|------|------|-------|
| `route_query` | Classifies query → vectorstore or web search | Gemini 1.5 Flash |
| `retrieve_documents` | Semantic search over local FAISS index | Gemini embedding-001 |
| `web_search` | Real-time search via DuckDuckGo (no key) | — |
| `grade_documents` | Checks if retrieved content is relevant | Gemini 1.5 Flash |
| `synthesize_answer` | Generates cited, grounded answer | Gemini 1.5 Flash |

### Conditional routing
- If router → `vectorstore` but grader says `not_relevant` → fallback to web search
- If router → `web_search` directly → skip retrieval
- Grounded synthesis always runs last

---

## Tech Stack

- **LangGraph** — stateful multi-node agent graph with conditional edges
- **LangChain** — LLM abstraction, prompt templates, output parsers
- **FAISS** — local vector store for semantic retrieval (no cloud dependency)
- **Google Gemini 1.5 Flash** — free LLM for routing/grading/synthesis
- **Gemini Embedding-001** — free embeddings (same API key)
- **DuckDuckGo Search** — free real-time web search, zero API key required
- **FastAPI** — async REST API with SSE streaming support
- **Python 3.11** — async/await throughout

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent/backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Get your free Gemini API key

1. Go to **https://aistudio.google.com/apikey**
2. Click "Create API key" — it's instant and free
3. Copy the key (starts with `AIza...`)

```bash
cp .env.example .env
# Open .env and set:
# GOOGLE_API_KEY=AIza...
```

That's the **only** key you need. DuckDuckGo search requires nothing.

### 3. Run the backend

```bash
cd backend
python server.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Open the frontend

Just open `frontend/index.html` in your browser. No build step needed.

---

## API Reference

### `POST /query`
Run a full research query.

```json
// Request
{ "query": "What is LangGraph?" }

// Response
{
  "query": "What is LangGraph?",
  "answer": "LangGraph is a library for building...",
  "sources": ["https://..."],
  "route": "web_search",
  "steps": ["router → web_search", "web_search", "grade → relevant", "synthesize"],
  "latency_ms": { "router": 312, "web_search": 890, "grader": 278, "synthesis": 1203 },
  "total_ms": 2690
}
```

### `POST /ingest/text`
Add text to the local knowledge base.

```json
{ "text": "Your document text...", "source_name": "my_paper.pdf" }
```

### `POST /ingest/file`
Upload a PDF/TXT/MD file (multipart form).

### `GET /vectorstore/stats`
Returns number of indexed vectors.

---

## Ingest your own documents

```python
# Python API
from ingest import ingest_text, ingest_file

# Index plain text
ingest_text("Your research paper content...", source_name="paper.txt")

# Index a PDF
with open("paper.pdf", "rb") as f:
    ingest_file(f.read(), "paper.pdf", ".pdf")
```

Or use the frontend's "Ingest Documents" button.

---

## Project Structure

```
research-agent/
├── backend/
│   ├── agent.py          # LangGraph state machine (core logic)
│   ├── ingest.py         # Chunking + FAISS indexing pipeline
│   ├── server.py         # FastAPI endpoints
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html        # Single-file UI (no build step)
└── README.md
```

---

## Key design decisions

**Why LangGraph over a simple chain?**
The routing + grading + fallback logic requires stateful branching. LangGraph's `StateGraph`
makes each decision point explicit and debuggable — the step trace in the UI shows exactly
which path was taken.

**Why FAISS over a hosted vector DB?**
Zero infrastructure for local use. The index persists to disk and can be swapped for
Pinecone/Weaviate by changing one class in `ingest.py`.

**Why grade retrieved documents?**
Without grading, queries about topics not in the knowledge base would get poor answers.
The grader node enables clean fallback to web search rather than hallucinating.

---

## License

MIT
