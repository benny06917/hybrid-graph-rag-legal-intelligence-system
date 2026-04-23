

import time
import logging
from dataclasses import dataclass
from typing import List

from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .config import cfg
from .retrieval import run_graph_rag

logger = logging.getLogger(__name__)

RELATIONAL_KEYWORDS = [
    "founded", "works at", "created", "collaborated", "part of",
    "related to", "won", "defeated", "studied under", "co-authored",
    "former", "previously", "joined", "left", "acquired",
]

VANILLA_PROMPT = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:
""")


@dataclass
class EvalResult:
    query: str
    answer: str
    latency_ms: float
    answer_length: int
    has_relational_info: bool
    mode: str


@dataclass
class ComparisonReport:
    query: str
    graph_rag: EvalResult
    vanilla_rag: EvalResult
    winner: str
    reason: str


def run_vanilla_rag(query: str, vectorstore: FAISS) -> EvalResult:
    t0 = time.perf_counter()
    llm = ChatGroq(
        model=cfg.llm.llm_model,
        temperature=cfg.llm.temperature,
        api_key=cfg.groq_api_key,
    )
    docs: List[Document] = vectorstore.similarity_search(query, k=cfg.retrieval.similarity_top_k)
    context = "\n\n".join(d.page_content for d in docs)
    chain = VANILLA_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    has_relational = any(kw in answer.lower() for kw in RELATIONAL_KEYWORDS)
    return EvalResult(
        query=query, answer=answer, latency_ms=latency_ms,
        answer_length=len(answer.split()),
        has_relational_info=has_relational, mode="vanilla_rag",
    )


def run_comparison(
    queries: List[str],
    graph: Neo4jGraph,
    vectorstore: FAISS,
) -> List[ComparisonReport]:
    reports = []
    for query in queries:
        logger.info(f"Evaluating: {query}")

        graph_response = run_graph_rag(query, graph, vectorstore)
        graph_relational = any(kw in graph_response.answer.lower() for kw in RELATIONAL_KEYWORDS)
        graph_result = EvalResult(
            query=query, answer=graph_response.answer,
            latency_ms=graph_response.latency_ms,
            answer_length=len(graph_response.answer.split()),
            has_relational_info=graph_relational, mode="graph_rag",
        )

        vanilla_result = run_vanilla_rag(query, vectorstore)

        graph_score   = int(graph_result.has_relational_info) * 2 + int(graph_result.answer_length > 25)
        vanilla_score = int(vanilla_result.has_relational_info) * 2 + int(vanilla_result.answer_length > 25)

        if graph_score > vanilla_score:
            winner = "GRAPH RAG"
            reason = "Neo4j graph traversal surfaced relational chains vector search missed"
        elif vanilla_score > graph_score:
            winner = "VANILLA RAG"
            reason = "Simple factual query — vector retrieval sufficient, lower latency"
        else:
            winner = "TIE"
            reason = "Comparable quality; Graph RAG adds explainability via Neo4j"

        reports.append(ComparisonReport(
            query=query, graph_rag=graph_result,
            vanilla_rag=vanilla_result, winner=winner, reason=reason,
        ))
    return reports


def format_report(reports: List[ComparisonReport]) -> str:
    lines = ["", "=" * 65, "   GRAPH RAG vs VANILLA RAG  —  EVALUATION REPORT", "=" * 65]
    graph_wins = vanilla_wins = ties = 0

    for i, r in enumerate(reports, 1):
        lines.append(f"\nQuery {i}: {r.query}")
        lines.append(f"  Winner  : {r.winner}")
        lines.append(f"  Reason  : {r.reason}")
        lines.append(
            f"  Graph RAG   — {r.graph_rag.answer_length:>3} words | "
            f"{r.graph_rag.latency_ms:>7.0f}ms | "
            f"relational={'yes' if r.graph_rag.has_relational_info else 'no'}"
        )
        lines.append(
            f"  Vanilla RAG — {r.vanilla_rag.answer_length:>3} words | "
            f"{r.vanilla_rag.latency_ms:>7.0f}ms | "
            f"relational={'yes' if r.vanilla_rag.has_relational_info else 'no'}"
        )
        lines.append(f"\n  Graph RAG answer:\n    {r.graph_rag.answer[:250].strip()}...")
        lines.append(f"\n  Vanilla answer:\n    {r.vanilla_rag.answer[:250].strip()}...")
        lines.append("-" * 65)

        if r.winner == "GRAPH RAG": graph_wins += 1
        elif r.winner == "VANILLA RAG": vanilla_wins += 1
        else: ties += 1

    lines.append(
        f"\nFINAL  —  Graph RAG: {graph_wins} wins  |  "
        f"Vanilla RAG: {vanilla_wins} wins  |  Ties: {ties}"
    )
    lines.append("=" * 65 + "\n")
    return "\n".join(lines)
