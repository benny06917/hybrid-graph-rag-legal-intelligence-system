
import os
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_groq import ChatGroq
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import cfg

logger = logging.getLogger(__name__)


TRIPLET_PROMPT = ChatPromptTemplate.from_template("""
Extract knowledge graph triplets from the text below.
Return ONLY lines in the format: Subject | predicate | Object
- Subject and Object must be named entities (people, orgs, technologies, concepts)
- Predicate should be a short verb phrase (founded, works_at, created, won, etc.)
- Extract up to {max_triplets} triplets maximum
- If no clear entities exist, return nothing

Text:
{text}

Triplets (one per line, format: Subject | predicate | Object):
""")


def _load_documents(data_dir: str) -> List[Document]:
    docs = []
    for filepath in Path(data_dir).glob("*.txt"):
        text = filepath.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": filepath.name}))
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    logger.info(f"Loaded {len(docs)} document(s)")
    return docs


def _chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.pipeline.chunk_size,
        chunk_overlap=cfg.pipeline.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunk(s)")
    return chunks


def _extract_triplets(chunk: Document, llm_chain) -> List[Tuple[str, str, str]]:
    try:
        raw = llm_chain.invoke({
            "text": chunk.page_content,
            "max_triplets": cfg.pipeline.max_triplets_per_chunk,
        })
        triplets = []
        for line in raw.strip().split("\n"):
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                s, p, o = parts
                if all(1 < len(x) < 80 for x in [s, p, o]):
                    triplets.append((s, p, o))
        return triplets
    except Exception as e:
        logger.warning(f"Triplet extraction failed: {e}")
        return []


def _write_to_neo4j(
    graph: Neo4jGraph,
    triplets: List[Tuple[str, str, str]],
    source_doc: Document,
) -> int:
 
    if not triplets:
        return 0

    nodes_map = {}
    relationships = []

    for subject, predicate, obj in triplets:
        # Create or reuse nodes
        if subject not in nodes_map:
            nodes_map[subject] = Node(id=subject, type="Entity")
        if obj not in nodes_map:
            nodes_map[obj] = Node(id=obj, type="Entity")

        # Normalise relation to UPPER_SNAKE_CASE for Neo4j convention
        rel_type = predicate.upper().replace(" ", "_").replace("-", "_")

        relationships.append(Relationship(
            source=nodes_map[subject],
            target=nodes_map[obj],
            type=rel_type,
        ))

    graph_doc = GraphDocument(
        nodes=list(nodes_map.values()),
        relationships=relationships,
        source=source_doc,
    )

    graph.add_graph_documents([graph_doc])
    return len(relationships)


def connect_neo4j() -> Neo4jGraph:
 
    try:
        graph = Neo4jGraph(
            url=cfg.neo4j.url,
            username=cfg.neo4j.username,
            password=cfg.neo4j.password,
        )
        # Test connection with a simple query
        graph.query("RETURN 1 as test")
        logger.info(f"Connected to Neo4j at {cfg.neo4j.url}")
        return graph
    except Exception as e:
        raise ConnectionError(
            f"\n\nCannot connect to Neo4j at {cfg.neo4j.url}\n"
            f"Make sure Neo4j Desktop is running and the database is started.\n"
            f"Check your password in .env matches Neo4j Desktop.\n"
            f"Original error: {e}\n"
        )


def build_graph_and_vectorstore(documents: List[Document]) -> Tuple[Neo4jGraph, FAISS]:
    llm = ChatGroq(
        model=cfg.llm.llm_model,
        temperature=cfg.llm.temperature,
        api_key=cfg.groq_api_key,
)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=cfg.openai_api_key)


    chunks = _chunk_documents(documents)
    triplet_chain = TRIPLET_PROMPT | llm | StrOutputParser()

    graph = connect_neo4j()
    total_triplets = 0

    logger.info("Extracting triplets and writing to Neo4j...")
    for i, chunk in enumerate(chunks):
        triplets = _extract_triplets(chunk, triplet_chain)
        written = _write_to_neo4j(graph, triplets, chunk)
        total_triplets += written
        if (i + 1) % 3 == 0:
            logger.info(f"  {i+1}/{len(chunks)} chunks | {total_triplets} triplets written to Neo4j")

    logger.info(f"Neo4j graph built: {total_triplets} total relationships")

    # Build FAISS vector store
    logger.info("Embedding chunks into FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(cfg.pipeline.persist_dir, exist_ok=True)
    vectorstore.save_local(cfg.pipeline.persist_dir)
    Path(cfg.pipeline.persist_dir, "ingestion_complete.flag").write_text("done")
    logger.info(f"FAISS persisted to {cfg.pipeline.persist_dir}")

    return graph, vectorstore


def load_or_build(force_rebuild: bool = False) -> Tuple[Neo4jGraph, FAISS]:

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=cfg.openai_api_key)

    faiss_file = Path(cfg.pipeline.persist_dir) / "index.faiss"
    flag_file  = Path(cfg.pipeline.persist_dir) / "ingestion_complete.flag"

    graph = connect_neo4j()

    if not force_rebuild and faiss_file.exists() and flag_file.exists():
        # Check Neo4j actually has data
        result = graph.query("MATCH (n) RETURN count(n) as count")
        node_count = result[0]["count"] if result else 0

        if node_count > 0:
            logger.info(f"Loading from cache — Neo4j has {node_count} nodes, FAISS exists")
            vectorstore = FAISS.load_local(
                cfg.pipeline.persist_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return graph, vectorstore

    logger.info("Building from scratch...")
    documents = _load_documents(cfg.pipeline.data_dir)
    return build_graph_and_vectorstore(documents)


def get_graph_stats(graph: Neo4jGraph) -> dict:
    try:
        nodes = graph.query("MATCH (n) RETURN count(n) as count")[0]["count"]
        rels  = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        return {"entities": nodes, "triplets": rels}
    except Exception:
        return {"entities": "n/a", "triplets": "n/a"}
