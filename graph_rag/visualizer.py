

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.graphs import Neo4jGraph

from .config import cfg

logger = logging.getLogger(__name__)


def get_triplets(graph: Neo4jGraph) -> List[Tuple[str, str, str]]:
    try:
        results = graph.query(
            "MATCH (s)-[r]->(o) RETURN s.id AS subject, type(r) AS predicate, o.id AS object LIMIT 1000"
        )
        return [(r["subject"], r["predicate"], r["object"]) for r in results if r["subject"] and r["object"]]
    except Exception as e:
        logger.warning(f"Could not fetch triplets: {e}")
        return []


def print_graph_summary(graph: Neo4jGraph) -> None:
    triplets = get_triplets(graph)

    if not triplets:
        print("  (No triplets in Neo4j yet — ingestion may not have run)")
        return

    entities = set()
    for s, _, o in triplets:
        entities.add(s)
        entities.add(o)

    print(f"\n  Entities : {len(entities)}")
    print(f"  Triplets : {len(triplets)}")
    print("\n  Sample triplets (first 15):")
    for s, p, o in triplets[:15]:
        print(f"    [{s}]  --{p}-->  [{o}]")
    if len(triplets) > 15:
        print(f"    ... and {len(triplets) - 15} more")

    print("\n  [TIP] Open http://localhost:7474 in your browser")
    print("        Run this Cypher to see the full visual graph:")
    print("        MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")


def export_edge_list(graph: Neo4jGraph, output_path: str = None) -> None:
    if output_path is None:
        output_path = str(Path(cfg.pipeline.persist_dir) / "graph_edges.tsv")

    triplets = get_triplets(graph)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("subject\tpredicate\tobject\n")
        for s, p, o in triplets:
            f.write(f"{s}\t{p}\t{o}\n")

    print(f"\n  Edge list exported → {output_path}")
