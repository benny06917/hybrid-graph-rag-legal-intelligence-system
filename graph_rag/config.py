
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GraphSchema:
    node_types: List[str] = field(default_factory=lambda: [
        "Person", "Organization", "Concept", "Technology", "Event",
    ])
    edge_types: List[str] = field(default_factory=lambda: [
        "FOUNDED", "WORKS_AT", "CREATED", "COLLABORATED_WITH",
        "PART_OF", "RELATED_TO", "WON", "DEFEATED", "STUDIED_UNDER",
    ])
    version: str = "1.0.0"


@dataclass
class Neo4jConfig:

    url: str = field(default_factory=lambda: os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687"))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", " "))


@dataclass
class LLMConfig:
    llm_model: str        = "llama-3.3-70b-versatile"
    embedding_model: str  = "text-embedding-3-small"
    temperature: float    = 0.0
    max_tokens: int       = 1024


@dataclass
class RetrievalConfig:
    graph_traversal_depth: int   = 2
    similarity_top_k: int        = 5
    max_triplets_in_context: int = 20


@dataclass
class PipelineConfig:
    data_dir: str               = "./data"
    persist_dir: str            = "./storage"
    chunk_size: int             = 512
    chunk_overlap: int          = 64
    max_triplets_per_chunk: int = 5


@dataclass
class Config:
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    neo4j: Neo4jConfig         = field(default_factory=Neo4jConfig)
    schema: GraphSchema        = field(default_factory=GraphSchema)
    llm: LLMConfig             = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig   = field(default_factory=PipelineConfig)


cfg = Config()
