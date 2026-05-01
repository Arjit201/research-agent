from __future__ import annotations
import traceback
import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import AgentState, build_graph, _embeddings
from ingest import ingest_file, ingest_text

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Agent API",
    description="LangGraph-powered research agent with RAG + web search",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    route: str
    steps: list[str]
    latency_ms: dict
    total_ms: float


class IngestTextRequest(BaseModel):
    text: str
    source_name: str = "manual_input"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_initial_state(query: str) -> AgentState:
    return AgentState(
        messages=[],
        query=query,
        route="web_search",
        documents=[],
        web_results=[],
        grade="not_relevant",
        answer="",
        sources=[],
        steps=[],
        latency_ms={},
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.5-flash"}


@app.get("/vectorstore/stats")
async def vectorstore_stats():
    vs_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")

    if not Path(vs_path).exists():
        return {"doc_count": 0, "indexed": False}

    try:
        from langchain_community.vectorstores import FAISS
        vs = FAISS.load_local(
            vs_path,
            _embeddings(),
            allow_dangerous_deserialization=True
        )
        return {"doc_count": vs.index.ntotal, "indexed": True}

    except Exception as e:
        return {"doc_count": 0, "indexed": False, "error": str(e)}

# ---------------------------------------------------------------------------
# MAIN QUERY
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    t0 = time.perf_counter()
    state = _build_initial_state(req.query)

    print("\n" + "="*60)
    print("NEW QUERY:", req.query)
    print("INITIAL STATE:", state)
    print("="*60)

    try:
        result = graph.invoke(state)

        # ✅ Ensure safe defaults
        result.setdefault("answer", "No answer generated.")
        result.setdefault("sources", [])
        result.setdefault("steps", [])
        result.setdefault("latency_ms", {})

    except Exception as e:
        print("\nFULL ERROR TRACE")
        traceback.print_exc()

        print("\nSTATE AT FAILURE:")
        print(state)

        raise HTTPException(500, f"Agent crashed: {str(e)}")

    total = round((time.perf_counter() - t0) * 1000, 1)

    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        sources=result["sources"],
        route=result.get("route", "unknown"),
        steps=result["steps"],
        latency_ms=result["latency_ms"],
        total_ms=total,
    )

# ---------------------------------------------------------------------------
# STREAMING
# ---------------------------------------------------------------------------

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    async def event_generator() -> AsyncGenerator[str, None]:
        state = _build_initial_state(req.query)
        t0 = time.perf_counter()

        def emit(type_: str, data: object):
            return f"data: {json.dumps({'type': type_, 'data': data})}\n\n"

        try:
            # Step streaming
            for event in graph.stream(state):
                for node_name, node_state in event.items():
                    steps = node_state.get("steps", [])
                    if steps:
                        yield emit("step", steps[-1])
                await asyncio.sleep(0)

            # Final result
            result = await asyncio.to_thread(graph.invoke, state)
            total = round((time.perf_counter() - t0) * 1000, 1)

            yield emit("answer", result.get("answer", ""))
            yield emit("meta", {
                "sources": result.get("sources", []),
                "route": result.get("route", ""),
                "steps": result.get("steps", []),
                "latency_ms": result.get("latency_ms", {}),
                "total_ms": total,
            })
            yield emit("done", None)

        except Exception as e:
            yield emit("error", str(e))
            yield emit("done", None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------

@app.post("/ingest/text")
async def ingest_text_endpoint(req: IngestTextRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    try:
        count = await asyncio.to_thread(
            ingest_text, req.text, req.source_name
        )
        return {"message": f"Indexed {count} chunks", "chunks": count}

    except Exception as e:
        raise HTTPException(500, f"Ingest error: {e}")


@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".md"}
    ext = Path(file.filename or "").suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'")

    content = await file.read()

    try:
        count = await asyncio.to_thread(
            ingest_file, content, file.filename or "upload", ext
        )
        return {"message": f"Indexed {count} chunks", "chunks": count}

    except Exception as e:
        raise HTTPException(500, f"Ingest error: {e}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)