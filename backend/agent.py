from __future__ import annotations

import os
import time
from google import genai
from typing import Annotated, Any, Literal, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    route: Literal["vectorstore", "web_search"]
    documents: list[dict]
    web_results: list[dict]
    grade: Literal["relevant", "not_relevant"]
    answer: str
    sources: list[str]
    steps: list[str]
    latency_ms: dict[str, float]

# ---------------------------------------------------------------------------
# LLM (FIXED)
# ---------------------------------------------------------------------------

def _llm():
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options={"api_version": "v1"}
    )

    def call_llm(input):
        if isinstance(input, dict):
            text = input.get("query") or input.get("context") or str(input)
        else:
            text = str(input)

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": text}]}]
        )

        return resp.candidates[0].content.parts[0].text

    return RunnableLambda(call_llm)

# ---------------------------------------------------------------------------
# Embeddings (REPLACED — stable)
# ---------------------------------------------------------------------------

def _embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Web Tool
# ---------------------------------------------------------------------------

def _web_tool():
    return DuckDuckGoSearchResults(num_results=5, output_format="list")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router.

Decide whether the query should go to:
- "vectorstore"
- "web_search"

Return ONLY valid JSON in this exact format:

{{"route": "vectorstore"}}

or

{{"route": "web_search"}}

DO NOT include anything else."""),
    ("human", "{query}"),
])
def route_query(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    chain = ROUTER_PROMPT | _llm() | JsonOutputParser()
    result = chain.invoke({"query": state["query"]})
    route = result.get("route", "web_search")

    return {
        **state,
        "route": route,
        "steps": state.get("steps", []) + [f"router → {route}"],
        "latency_ms": {"router": round((time.perf_counter() - t0) * 1000, 1)},
    }

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def retrieve_documents(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    vs_path = "./vectorstore"
    docs = []

    if os.path.exists(vs_path):
        vs = FAISS.load_local(vs_path, _embeddings(), allow_dangerous_deserialization=True)
        raw = vs.similarity_search_with_score(state["query"], k=4)

        docs = [
            {
                "content": d.page_content,
                "source": d.metadata.get("source", "local"),
                "score": float(s),
            }
            for d, s in raw
        ]

    return {
        **state,
        "documents": docs,
        "steps": state["steps"] + ["retrieve"],
        "latency_ms": {"retrieval": round((time.perf_counter() - t0) * 1000, 1)},
    }

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def web_search(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    results = _web_tool().invoke(state["query"])

    parsed = [
        {
            "content": r.get("snippet", ""),
            "source": r.get("link", ""),
        }
        for r in results
    ]

    return {
        **state,
        "web_results": parsed,
        "steps": state["steps"] + ["web"],
        "latency_ms": {"web": round((time.perf_counter() - t0) * 1000, 1)},
    }

# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document relevance grader.

Given a query and some retrieved documents, decide if they are relevant.

Return ONLY valid JSON in this format:

{{"grade": "relevant"}}

or

{{"grade": "not_relevant"}}

DO NOT include anything else."""),
    ("human", "Query: {query}\nDocs:\n{docs}")
])

def grade_documents(state: AgentState) -> AgentState:
    docs = state.get("documents", []) + state.get("web_results", [])
    text = "\n".join(d["content"][:300] for d in docs[:5])

    result = (GRADER_PROMPT | _llm() | JsonOutputParser()).invoke({
        "query": state["query"],
        "docs": text
    })

    return {
        **state,
        "grade": result.get("grade", "not_relevant"),
        "steps": state["steps"] + ["grade"],
    }

# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using provided context. Cite sources."),
    ("human", "Query: {query}\nContext:\n{context}")
])

def synthesize_answer(state: AgentState) -> AgentState:
    docs = state.get("documents", []) + state.get("web_results", [])

    context = "\n\n".join(
        f"[{i+1}] {d['content'][:500]}"
        for i, d in enumerate(docs[:5])
    ) or "No context"

    answer = (SYNTH_PROMPT | _llm() | StrOutputParser()).invoke({
        "query": state["query"],
        "context": context
    })

    return {
        **state,
        "answer": answer,
        "messages": state["messages"] + [AIMessage(content=answer)],
        "steps": state["steps"] + ["synthesize"],
    }

# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def decide_retrieval_path(state: AgentState):
    return state["route"]

def decide_after_grading(state: AgentState):
    if state["grade"] == "relevant":
        return "synthesize_answer"
    if state.get("web_results"):
        return "synthesize_answer"
    return "web_search"

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route_query", route_query)
    g.add_node("retrieve_documents", retrieve_documents)
    g.add_node("web_search", web_search)
    g.add_node("grade_documents", grade_documents)
    g.add_node("synthesize_answer", synthesize_answer)

    g.add_edge(START, "route_query")
    g.add_conditional_edges("route_query", decide_retrieval_path)
    g.add_edge("retrieve_documents", "grade_documents")
    g.add_edge("web_search", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_after_grading)
    g.add_edge("synthesize_answer", END)

    return g.compile()

graph = build_graph()