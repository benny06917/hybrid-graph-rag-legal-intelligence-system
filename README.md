# Graph RAG 

> Knowledge graphs meet vector search — a Hybrib Retrieval-Augmented Generation pipeline that reasons over *relationships*, not just text chunks.

---

## Overview

Standard RAG is powerful but blind to context. Ask *"How are Hinton and LeCun connected?"* and a vanilla vector search will retrieve isolated facts. It won't trace the path between two people through shared awards, institutions, or ideas.

**Graph RAG** fixes this. It combines:

- 🔗 **Neo4j** — a local knowledge graph that stores entities and their relationships as first-class data
- 🧠 **LangChain + OpenAI** — LLM-powered triplet extraction at ingestion, and grounded answer generation at query time
- ⚡ **FAISS** — a local vector index for fast semantic retrieval alongside graph traversal
- 🔍 **k-hop Cypher traversal** — multi-hop graph queries that surface indirect connections vanilla RAG simply cannot find

The result: answers that are factually grounded *and* relationally aware.

---

## Objective

This project demonstrates a **Hybrid Graph RAG architecture** with three goals:

1. **Build a working knowledge graph from raw text** — documents are chunked, entity relationships are extracted as `(subject, predicate, object)` triplets, and written into Neo4j alongside a FAISS vector index.

2. **Answer multi-hop questions** — at query time, the pipeline runs semantic retrieval *and* Cypher graph traversal in parallel, then fuses both result sets before generating a grounded response.

3. **Benchmark against vanilla RAG** — a built-in evaluation mode (`--eval`) runs both pipelines on the same questions and scores them side-by-side, making the quality gap between graph-aware and chunk-only retrieval concrete and measurable.


## Folder Structure

```
graph_rag_neo4j/              ← cd into this before running anything
│
├── main.py                   ← entrypoint, run this
├── requirements.txt
├── .env.example              ← copy to .env and fill in
│
├── data/
│   ├── legal_research.txt       ← sample data (swap with your own)
│   
│
├── graph_rag/                ← the pipeline package
│   ├── __init__.py
│   ├── config.py             ← all config + Neo4j connection settings
│   ├── ingestion.py          ← chunking, triplet extraction, Neo4j write, FAISS build
│   ├── retrieval.py          ← Cypher k-hop traversal + FAISS + LLM generation
│   ├── evaluation.py         ← Graph RAG vs Vanilla RAG comparison
│   └── visualizer.py         ← graph summary + edge list export
│
└── storage/                  ← auto-created on first run
    ├── index.faiss           ← FAISS vector index
    ├── index.pkl
    ├── ingestion_complete.flag
    └── graph_edges.tsv       ← exportable edge list
```

## Setup — Step by Step

### Step 1 — Neo4j Desktop
1. Download from https://neo4j.com/download
2. Install and open Neo4j Desktop
3. Click "New Project" → "Add" → "Local DBMS"
4. Set a password (use `password123` or update `.env`)
5. Click **Start** — the green dot means it's running
6. Open browser at http://localhost:7474 to confirm

### Step 2 — Python environment
```bash
cd graph_rag_neo4j
pip install -r requirements.txt
```

### Step 3 — Environment variables
```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and NEO4J_PASSWORD
```
Or export directly:
```bash
export OPENAI_API_KEY=sk-...
export NEO4J_PASSWORD=your-neo4j-password
```

### Step 4 — Run
```bash
python main.py                  # interactive Q&A
python main.py --eval           # Graph RAG vs Vanilla RAG scorecard
python main.py --show-graph     # print graph summary
python main.py --query "How are Hinton and LeCun connected?"
python main.py --rebuild        # re-ingest (after changing data/)
```

## What Connects to What

| Thing | Where it runs | Cost |
|---|---|---|
| OpenAI API | cloud (api.openai.com) | ~$0.10–0.30 per full run |
| Neo4j | local (localhost:7687) | free |
| FAISS | local (storage/ folder) | free |

## Neo4j Visual Graph (demo tip)

Once ingestion runs, open http://localhost:7474 and run:
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```
This shows the full interactive knowledge graph — best visual for the demo.

## Best Demo Queries

```
How are Geoffrey Hinton and Yann LeCun connected?
Who founded OpenAI and what did they create?
Which award did Hinton, LeCun and Bengio win together?
What is the relationship between Anthropic and OpenAI?
What is GraphRAG and how does it improve on standard RAG?
```

