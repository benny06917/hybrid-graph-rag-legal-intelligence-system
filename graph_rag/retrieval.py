
import logging
import time
from dataclasses import dataclass
from typing import List

from langchain_groq import ChatGroq
#from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .config import cfg

logger = logging.getLogger(__name__)


ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
Extract named entities (people, organisations, technologies, concepts) from this query.
Return ONLY a comma-separated list of entity names. Nothing else.

Query: {query}

Entities:
""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions using a knowledge graph and document context.

KNOWLEDGE GRAPH CONTEXT (entity relationships from Neo4j):
{graph_context}

DOCUMENT CONTEXT (relevant passages):
{vector_context}

Question: {question}

Instructions:
- Use the knowledge graph context to reason about relationships between entities
- Use the document context for supporting detail
- If the graph reveals a multi-hop connection, explain it clearly
- Be concise and factual

Answer:
""")


@dataclass
class ParsedQuery:
    original: str
    entities: List[str]


@dataclass
class RAGResponse:
    question: str
    answer: str
    graph_context: str
    vector_context: str
    entities: List[str]
    latency_ms: float


def parse_query(query: str, llm: ChatGroq) -> ParsedQuery:
    chain = ENTITY_EXTRACTION_PROMPT | llm | StrOutputParser()
    try:
        raw = chain.invoke({"query": query})
        entities = [e.strip() for e in raw.split(",") if e.strip() and len(e.strip()) > 1]
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        entities = []
    logger.info(f"Query entities: {entities}")
    return ParsedQuery(original=query, entities=entities)


def retrieve_graph_context(
    graph: Neo4jGraph,
    entities: List[str],
    depth: int = None,
) -> str:

    depth = depth or cfg.retrieval.graph_traversal_depth

    if not entities:
        return "No entities found in query."

    facts = set()

    for entity in entities:
        try:
            # Cypher k-hop query — case-insensitive match on entity id
            cypher = """
            MATCH (e)
            WHERE toLower(e.id) CONTAINS toLower($entity)
            OPTIONAL MATCH (e)-[r1]-(n1)
            OPTIONAL MATCH (n1)-[r2]-(n2)
            RETURN e.id AS entity,
                   type(r1) AS rel1, n1.id AS neighbor1,
                   type(r2) AS rel2, n2.id AS neighbor2
            LIMIT 50
            """
            results = graph.query(cypher, params={"entity": entity})

            for row in results:
                if row.get("rel1") and row.get("neighbor1"):
                    facts.add(f"{row['entity']} {row['rel1']} {row['neighbor1']}")
                if row.get("rel2") and row.get("neighbor2") and row.get("neighbor1"):
                    facts.add(f"{row['neighbor1']} {row['rel2']} {row['neighbor2']}")

        except Exception as e:
            logger.warning(f"Neo4j query failed for '{entity}': {e}")

    if not facts:
        return "No graph relationships found for query entities."

    trimmed = list(facts)[:cfg.retrieval.max_triplets_in_context]
    logger.info(f"Graph context: {len(trimmed)} facts from Neo4j")
    return "\n".join(f"  • {fact}" for fact in trimmed)


def retrieve_vector_context(vectorstore: FAISS, query: str) -> str:
    try:
        docs: List[Document] = vectorstore.similarity_search(
            query, k=cfg.retrieval.similarity_top_k
        )
        if not docs:
            return "No relevant document passages found."
        passages = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            passages.append(f"  [{i}] ({source})\n  {doc.page_content.strip()}")
        logger.info(f"Vector context: {len(passages)} passages")
        return "\n\n".join(passages)
    except Exception as e:
        logger.warning(f"Vector retrieval failed: {e}")
        return "Vector retrieval unavailable."


def rerank_facts(facts_text: str, query: str) -> str:
    lines = [l.strip() for l in facts_text.split("\n") if l.strip() and l.strip() != "No graph relationships found for query entities."]
    if not lines:
        return facts_text

    query_tokens = set(query.lower().split())

    def _score(line: str) -> float:
        return len(query_tokens & set(line.lower().split())) / max(len(query_tokens), 1)

    reranked = sorted(lines, key=_score, reverse=True)
    return "\n".join(reranked)


def run_graph_rag(
    query: str,
    graph: Neo4jGraph,
    vectorstore: FAISS,
) -> RAGResponse:
    t0 = time.perf_counter()

    llm = ChatGroq(
        model=cfg.llm.llm_model,
        temperature=cfg.llm.temperature,
        api_key=cfg.groq_api_key,
        )

    parsed       = parse_query(query, llm)
    graph_ctx    = retrieve_graph_context(graph, parsed.entities)
    graph_ctx    = rerank_facts(graph_ctx, query)
    vector_ctx   = retrieve_vector_context(vectorstore, query)

    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "graph_context": graph_ctx,
        "vector_context": vector_ctx,
        "question": query,
    })

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"Query completed in {latency_ms}ms")

    return RAGResponse(
        question=query,
        answer=answer,
        graph_context=graph_ctx,
        vector_context=vector_ctx,
        entities=parsed.entities,
        latency_ms=latency_ms,
    )
