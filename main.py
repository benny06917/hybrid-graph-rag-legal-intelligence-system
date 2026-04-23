
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

console = Console()


EVAL_QUERIES = [
    "Who founded OpenAI and what did they create?",
    "How are Geoffrey Hinton and Yann LeCun connected?",
    "Which award did Hinton, LeCun and Bengio win together?",
    "What is the relationship between Anthropic and OpenAI?",
    "What is GraphRAG and how does it improve on standard RAG?",
]



def main():
    parser = argparse.ArgumentParser(description="Graph RAG — LangChain + Neo4j")
    parser.add_argument("--eval",       action="store_true", help="Run evaluation comparison")
    parser.add_argument("--show-graph", action="store_true", help="Print graph summary and exit")
    parser.add_argument("--query",      type=str,            help="Single query mode")
    parser.add_argument("--rebuild",    action="store_true", help="Clear graph + FAISS, re-ingest")
    args = parser.parse_args()

    #console.print(BANNER)
    console.print(Panel(
        "[bold]Graph RAG[/bold] · LangChain · [bold cyan]Neo4j[/bold cyan] · FAISS · OpenAI\n"
        "[dim]Hybrid: Cypher k-hop traversal + vector similarity · LCEL chains[/dim]",
        border_style="cyan",
    ))


    from graph_rag.config import cfg

    if args.rebuild:
        if os.path.exists(cfg.pipeline.persist_dir):
            shutil.rmtree(cfg.pipeline.persist_dir)
            console.print("  [yellow]✓ Cleared FAISS storage[/yellow]")
        # Connect and clear Neo4j
        try:
            from graph_rag.ingestion import connect_neo4j
            graph = connect_neo4j()
            graph.query("MATCH (n) DETACH DELETE n")
            console.print("  [yellow]✓ Cleared Neo4j graph[/yellow]")
        except Exception as e:
            console.print(f"  [red]Could not clear Neo4j: {e}[/red]")

    # start the ingestion
    console.rule("[bold cyan]① Ingestion")
    console.print("  Connecting to Neo4j and loading documents...")

    from graph_rag import (
        load_or_build, get_graph_stats,
        run_graph_rag,
        run_comparison, format_report,
        print_graph_summary, export_edge_list,
    )

    try:
        graph, vectorstore = load_or_build(force_rebuild=args.rebuild)
    except ConnectionError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)

    stats = get_graph_stats(graph)
    console.print(
        f"  [green]✓[/green] Neo4j connected — "
        f"[bold]{stats['entities']}[/bold] nodes, "
        f"[bold]{stats['triplets']}[/bold] relationships"
    )

    # Graph summary
    console.rule("[bold cyan]② Knowledge Graph")
    print_graph_summary(graph)
    export_edge_list(graph)

    if args.show_graph:
        return

    console.rule("[bold cyan]③ Retrieval Engine")
    console.print(
        f"  Mode: [bold]Hybrid[/bold] — "
        f"Neo4j Cypher traversal (depth={cfg.retrieval.graph_traversal_depth}) "
        f"+ FAISS vector search (top_k={cfg.retrieval.similarity_top_k})"
    )
    console.print("  Stack: [cyan]LangChain LCEL[/cyan] · Neo4jGraph · FAISS")
    console.print("  [green]✓[/green] Pipeline ready")

    # for evaluation
    if args.eval:
        console.rule("[bold cyan]④ Evaluation — Graph RAG vs Vanilla RAG")
        console.print(f"  Running {len(EVAL_QUERIES)} queries through both pipelines...\n")
        reports = run_comparison(EVAL_QUERIES, graph, vectorstore)
        print(format_report(reports))
        return

    # to use for only single query
    if args.query:
        console.rule("[bold cyan]④ Generation")
        with console.status("[cyan]Cypher traversal + vector search + generation..."):
            result = run_graph_rag(args.query, graph, vectorstore)
        console.print(Panel(
            result.answer,
            title=f"[bold cyan]Q:[/bold cyan] {args.query}",
            border_style="green",
        ))
        console.print(f"\n  [dim]Entities: {', '.join(result.entities) or 'none'}[/dim]")
        console.print(f"  [dim]Latency: {result.latency_ms}ms[/dim]")
        return

    console.rule("[bold cyan]④ Interactive Q&A")
    console.print("  Type a question and press Enter. Type [bold]exit[/bold] to quit.\n")
    console.print("  [dim]Good demo queries:[/dim]")
    for q in EVAL_QUERIES[:3]:
        console.print(f"    [cyan]→[/cyan] {q}")
    console.print()

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Bye![/dim]")
            break

        with console.status("[cyan]Querying Neo4j + FAISS + generating..."):
            result = run_graph_rag(query, graph, vectorstore)

        console.print(Panel(
            result.answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print(
            f"  [dim]↳ Entities: {', '.join(result.entities) or 'none'} · "
            f"Latency: {result.latency_ms}ms[/dim]\n"
        )


if __name__ == "__main__":
    main()
