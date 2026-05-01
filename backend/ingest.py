from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------------------------------------------------------
# Embeddings (FIXED — local model, no API issues)
# ---------------------------------------------------------------------------

def _embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Vectorstore helpers
# ---------------------------------------------------------------------------

def _load_or_create_vs(emb) -> FAISS | None:
    p = Path(VECTORSTORE_PATH)
    if p.exists():
        return FAISS.load_local(str(p), emb, allow_dangerous_deserialization=True)
    return None


def _save_vs(vs: FAISS) -> None:
    Path(VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)
    vs.save_local(VECTORSTORE_PATH)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str) -> List[Document]:
    chunks = _splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": source}) for c in chunks]

# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        raise RuntimeError("Install pypdf: pip install pypdf")

# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def ingest_text(text: str, source_name: str = "manual") -> int:
    emb = _embeddings()
    docs = _chunk_text(text, source_name)

    if not docs:
        return 0

    vs = _load_or_create_vs(emb)

    if vs is None:
        vs = FAISS.from_documents(docs, emb)
    else:
        vs.add_documents(docs)

    _save_vs(vs)
    return len(docs)


def ingest_file(content: bytes, filename: str, ext: str) -> int:
    if ext == ".pdf":
        text = _extract_pdf_text(content)
    else:
        text = content.decode("utf-8", errors="ignore")

    return ingest_text(text, filename)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = ingest_text(
        "LangGraph is a framework for building stateful multi-agent applications.",
        source_name="test_doc",
    )
    print(f"Indexed {n} chunks")