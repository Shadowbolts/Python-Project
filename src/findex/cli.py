import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from findex.crawler.crawl_cmd import run_crawl_cli
from findex.index import build_index
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import search
from findex.snippet import snippet_for_doc
from findex.store import load_index, save_index

app = typer.Typer(
    name="findex",
    help="A full-text search engine for text corpora.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def configure_logging(verbose: int) -> None:
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)."),
    ] = 0,
) -> None:
    configure_logging(verbose)


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="Path to corpus directory or file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output path for saved index.")] = Path("index.pkl"),
    positions: Annotated[bool, typer.Option("--positions", help="Record token positions for phrase queries.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Limit number of documents to index.")] = None,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of parallel workers.")] = 1,
    executor: Annotated[
        Literal["serial", "threads", "processes"],
        typer.Option("--executor", "-e", help="Execution backend (serial, threads, processes)."),
    ] = "processes",
    format: Annotated[Literal["pickle", "json"] | None, typer.Option("--format", help="Serialization format.")] = None,
) -> None:
    if not corpus.exists():
        err_console.print(f"[bold red]Error:[/bold red] Corpus path '{corpus}' does not exist.")
        raise typer.Exit(code=1)

    with console.status(f"[bold green]Indexing {corpus}...[/bold green]"):
        try:
            idx = build_index(
                corpus,
                record_positions=positions,
                limit=limit,
                workers=workers,
                executor=executor,
            )
        except Exception as e:
            err_console.print(f"[bold red]Indexing failed:[/bold red] {e}")
            raise typer.Exit(code=1) from e

        save_index(idx, out, format=format)

    console.print(
        f"[bold green]Successfully indexed[/bold green] {idx.num_docs} documents "
        f"({len(idx)} unique terms) via [yellow]{executor}[/yellow] ({workers} worker{'s' if workers > 1 else ''}) -> [cyan]{out}[/cyan]"
    )


@app.command()
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index file.")],
    query: Annotated[str, typer.Argument(help="Search query string.")],
    k: Annotated[int, typer.Option("--k", "-k", help="Number of results to return.")] = 10,
    scorer: Annotated[Literal["bm25", "tfidf"], typer.Option("--scorer", help="Ranking algorithm.")] = "bm25",
    as_json: Annotated[bool, typer.Option("--json", help="Output raw JSON results for piping.")] = False,
    snippets: Annotated[bool, typer.Option("--snippets", help="Display snippets with highlighted query terms.")] = True,
) -> None:
    if not index_path.is_file():
        err_console.print(f"[bold red]Error:[/bold red] Index file '{index_path}' not found.")
        raise typer.Exit(code=1)

    try:
        idx = load_index(index_path)
    except (OSError, ValueError, TypeError) as e:
        err_console.print(f"[bold red]Error loading index:[/bold red] {e}")
        raise typer.Exit(code=1)

    active_scorer: Scorer = BM25() if scorer == "bm25" else TfIdf()
    results = search(query, idx, scorer=active_scorer, k=k)

    if as_json:
        for r in results:
            print(json.dumps({"doc_id": r.doc_id, "score": r.score, "title": r.title, "path": r.path}))
        return

    if not results:
        console.print("[yellow]No matching documents found.[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}' ({scorer.upper()})")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Title", style="bold white")
    table.add_column("Snippet", style="green")

    query_terms = query.split()
    for i, r in enumerate(results, start=1):
        snip = snippet_for_doc(r.path, query_terms) if snippets else ""
        table.add_row(str(i), f"{r.score:.4f}", r.title, snip)

    console.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index file.")],
) -> None:
    if not index_path.is_file():
        err_console.print(f"[bold red]Error:[/bold red] Index file '{index_path}' not found.")
        raise typer.Exit(code=1)

    idx = load_index(index_path)
    table = Table(title=f"Index Statistics: {index_path.name}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Documents", str(idx.num_docs))
    table.add_row("Unique Terms (Vocabulary)", str(len(idx)))
    table.add_row("Average Document Length", f"{idx.avg_doc_length:.1f} tokens")

    total_postings = sum(len(p) for p in idx.postings.values())
    table.add_row("Total Postings", str(total_postings))

    console.print(table)


@app.command(name="crawl")
def crawl_cmd(
    seed: Annotated[str, typer.Argument(help="Seed URL to start crawling from.")],
    max_pages: Annotated[int, typer.Option("--max-pages", "-n", help="Maximum pages to crawl.")] = 200,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Global concurrent workers.")] = 10,
    per_host: Annotated[int, typer.Option("--per-host", help="Max concurrent requests per host.")] = 2,
    min_delay: Annotated[float, typer.Option("--min-delay", help="Minimum delay between requests to same host (s).")] = 0.05,
    allowed_domain: Annotated[
        list[str] | None,
        typer.Option("--allowed-domain", "-d", help="Restrict crawl to specific domain(s)."),
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL file path.")] = Path("crawl.jsonl"),
    log_file: Annotated[Path, typer.Option("--log", help="Log file path.")] = Path("crawl.log"),
) -> None:
    """Crawl web pages asynchronously and stream results to JSONL."""
    pages_count, elapsed = asyncio.run(
        run_crawl_cli(
            [seed],
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=min_delay,
            allowed_domains=allowed_domain,
            out_path=out,
            log_path=log_file,
            console=console,
        )
    )
    rate = pages_count / elapsed if elapsed > 0 else 0.0
    console.print(
        f"[bold green]Crawled[/bold green] {pages_count} pages in {elapsed:.2f}s "
        f"({rate:.2f} pages/s) -> [cyan]{out}[/cyan] (log: [cyan]{log_file}[/cyan])"
    )


@app.command(name="serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Bind port.")] = 8000,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of uvicorn worker processes.")] = 1,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload for development.")] = False,
) -> None:
    """Start the FastAPI web search server via Uvicorn."""
    import uvicorn

    uvicorn.run(
        "findex.web:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        proxy_headers=True,
    )


if __name__ == "__main__":
    app()
