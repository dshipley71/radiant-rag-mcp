"""
Display and postprocessing utilities for Radiant Agentic RAG.

Provides formatted output using Rich console for debugging and monitoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from radiant_rag_mcp.utils.metrics import RunMetrics
from radiant_rag_mcp.storage.base import StoredDoc

logger = logging.getLogger(__name__)

# Global console instance
console = Console()


def format_latency(ms: Optional[float]) -> str:
    """
    Format latency in human-readable form.
    
    Converts a millisecond value into a string representation that uses
    milliseconds for values under 1 second and seconds for larger values.
    
    Args:
        ms: Latency value in milliseconds, or None if unavailable.
        
    Returns:
        Human-readable latency string:
        - "N/A" if ms is None
        - "{value}ms" for values < 1000ms (e.g., "42.5ms")
        - "{value}s" for values >= 1000ms (e.g., "1.50s")
        
    Examples:
        >>> format_latency(None)
        'N/A'
        >>> format_latency(42.5)
        '42.5ms'
        >>> format_latency(1500)
        '1.50s'
    """
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def display_step_metrics(metrics: RunMetrics) -> None:
    """
    Display step-by-step metrics in a table.
    
    Args:
        metrics: RunMetrics to display
    """
    table = Table(title="Pipeline Steps", show_header=True, header_style="bold cyan")
    table.add_column("Step", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Details")

    for step in metrics.steps:
        status = "[green]✓[/green]" if step.ok else "[red]✗[/red]"
        latency = format_latency(step.latency_ms)
        
        details_parts = []
        for key, value in step.extra.items():
            if isinstance(value, (int, float)):
                details_parts.append(f"{key}={value}")
            elif isinstance(value, str) and len(value) < 50:
                details_parts.append(f"{key}={value}")
        
        details = ", ".join(details_parts) if details_parts else ""
        
        table.add_row(step.name, status, latency, details)

    console.print(table)

    # Show warnings and degraded features
    if metrics.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in metrics.warnings:
            console.print(f"  • {warning}")

    if metrics.degraded_features:
        console.print("\n[orange1]Degraded Features:[/orange1]")
        for feature in metrics.degraded_features:
            console.print(f"  • {feature}")


def display_retrieval_results(
    docs: List[Tuple[StoredDoc, float]],
    title: str = "Retrieved Documents",
    max_content_chars: int = 200,
) -> None:
    """
    Display retrieval results in a formatted table.
    
    Args:
        docs: List of (document, score) tuples
        title: Table title
        max_content_chars: Maximum content characters to display
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Source", style="cyan", width=30)
    table.add_column("Content Preview")

    for i, (doc, score) in enumerate(docs, start=1):
        source = doc.meta.get("source_path", doc.doc_id[:20])
        if isinstance(source, str) and len(source) > 30:
            source = "..." + source[-27:]
        
        content = doc.content[:max_content_chars]
        if len(doc.content) > max_content_chars:
            content += "..."
        
        # Clean up content for display
        content = content.replace("\n", " ").replace("\t", " ")
        
        table.add_row(str(i), f"{score:.4f}", str(source), content)

    console.print(table)


def display_query_processing(
    original: str,
    decomposed: List[str],
    rewrites: List[Tuple[str, str]],
    expansions: List[str],
) -> None:
    """
    Display query processing results.
    
    Args:
        original: Original query
        decomposed: Decomposed sub-queries
        rewrites: List of (before, after) rewrite tuples
        expansions: Expansion terms
    """
    tree = Tree("[bold]Query Processing[/bold]")
    
    # Original
    tree.add(f"[cyan]Original:[/cyan] {original}")
    
    # Decomposition
    if decomposed and len(decomposed) > 1:
        decomp_branch = tree.add("[yellow]Decomposed:[/yellow]")
        for q in decomposed:
            decomp_branch.add(q)
    
    # Rewrites
    if rewrites:
        rewrite_branch = tree.add("[green]Rewrites:[/green]")
        for before, after in rewrites:
            if before != after:
                rewrite_branch.add(f"{before} → {after}")
    
    # Expansions
    if expansions:
        exp_text = ", ".join(expansions)
        tree.add(f"[magenta]Expansions:[/magenta] {exp_text}")
    
    console.print(tree)


def display_answer(
    query: str,
    answer: str,
    source_docs: Optional[List[Tuple[Any, float]]] = None,
    critic_notes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Display the final answer with source references and optional critique.
    
    Args:
        query: Original query
        answer: Generated answer
        source_docs: Optional list of (StoredDoc, score) tuples
        critic_notes: Optional critique results
    """
    # Query panel
    console.print(Panel(query, title="[bold blue]Query[/bold blue]", border_style="blue"))
    
    # Answer panel
    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))
    
    # Source documents table
    if source_docs:
        # Normalize scores if we have scored docs
        # Find max score for normalization
        max_score = max(score for _, score in source_docs) if source_docs else 1.0
        if max_score == 0:
            max_score = 1.0
        
        doc_table = Table(
            title="📚 Source Documents",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )
        doc_table.add_column("#", style="cyan", justify="center", width=4)
        doc_table.add_column("Source", style="blue", max_width=25)
        doc_table.add_column("Page", justify="center", width=5)
        doc_table.add_column("Score", justify="center", width=8)
        doc_table.add_column("Content Preview", style="dim", max_width=55)
        
        for i, item in enumerate(source_docs, start=1):
            # Handle both (doc, score) tuples and plain docs
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = 0.0
            
            # Get source path
            source = doc.meta.get("source_path", doc.meta.get("source", "unknown"))
            if "/" in source:
                source = source.split("/")[-1]
            
            page = str(doc.meta.get("page_number", "-"))
            
            # Normalize score to 0-100
            normalized_score = (score / max_score) * 100 if max_score > 0 else 0
            
            # Score color based on normalized value
            if normalized_score >= 70:
                score_str = f"[green]{normalized_score:.0f}%[/green]"
            elif normalized_score >= 40:
                score_str = f"[yellow]{normalized_score:.0f}%[/yellow]"
            else:
                score_str = f"[red]{normalized_score:.0f}%[/red]"
            
            # Content preview (first 120 chars)
            preview = doc.content[:120].replace("\n", " ").strip()
            if len(doc.content) > 120:
                preview += "..."
            
            doc_table.add_row(f"[{i}]", source, page, score_str, preview)
        
        console.print(doc_table)
    
    # Critique panel
    if critic_notes:
        ok = critic_notes.get("ok", True)
        status_color = "green" if ok else "yellow"
        
        critique_text = Text()
        
        # Scores
        for score_name in ["relevance_score", "faithfulness_score", "coverage_score"]:
            if score_name in critic_notes:
                score = critic_notes[score_name]
                score_color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
                critique_text.append(f"{score_name.replace('_', ' ').title()}: ")
                critique_text.append(f"{score}/10", style=score_color)
                critique_text.append("  ")
        
        # Issues
        issues = critic_notes.get("issues", [])
        if issues:
            critique_text.append("\n\nIssues:\n", style="bold")
            for issue in issues:
                critique_text.append(f"  • {issue}\n")
        
        # Suggestions
        suggestions = critic_notes.get("suggested_improvements", [])
        if suggestions:
            critique_text.append("\nSuggestions:\n", style="bold")
            for suggestion in suggestions:
                critique_text.append(f"  • {suggestion}\n")
        
        console.print(Panel(
            critique_text,
            title=f"[bold {status_color}]Critique[/bold {status_color}]",
            border_style=status_color,
        ))


def display_run_summary(metrics: RunMetrics) -> None:
    """
    Display a summary of the pipeline run.
    
    Args:
        metrics: RunMetrics from the run
    """
    # Build summary text
    summary = Text()
    
    summary.append("Run ID: ", style="bold")
    summary.append(f"{metrics.run_id}\n")
    
    summary.append("Total Time: ", style="bold")
    summary.append(f"{format_latency(metrics.total_latency_ms)}\n")
    
    summary.append("Steps: ", style="bold")
    total_steps = len(metrics.steps)
    successful = sum(1 for s in metrics.steps if s.ok)
    summary.append(f"{successful}/{total_steps} successful ")
    summary.append(f"({metrics.success_rate:.0%})\n", style="green" if metrics.success_rate == 1.0 else "yellow")
    
    # Failed steps
    failed = metrics.failed_steps
    if failed:
        summary.append("\nFailed Steps:\n", style="bold red")
        for step in failed:
            summary.append(f"  • {step.name}: {step.error}\n", style="red")
    
    console.print(Panel(summary, title="[bold]Run Summary[/bold]"))


def display_config_summary(config_dict: Dict[str, Any]) -> None:
    """
    Display configuration summary.
    
    Args:
        config_dict: Configuration as dictionary
    """
    tree = Tree("[bold]Configuration[/bold]")
    
    def add_dict_to_tree(d: Dict[str, Any], branch: Tree) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                sub_branch = branch.add(f"[cyan]{key}[/cyan]")
                add_dict_to_tree(value, sub_branch)
            else:
                branch.add(f"[dim]{key}:[/dim] {value}")
    
    add_dict_to_tree(config_dict, tree)
    console.print(tree)


def print_header(text: str) -> None:
    """Print a styled header."""
    console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
    console.print(f"[bold blue]{text.center(60)}[/bold blue]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")


def print_separator() -> None:
    """Print a separator line."""
    console.print("[dim]" + "-" * 60 + "[/dim]")


def print_success(message: str, title: Optional[str] = None) -> None:
    """Print a success message."""
    if title:
        console.print(f"[bold green]{title}[/bold green]")
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str, title: Optional[str] = None) -> None:
    """Print an error message."""
    if title:
        console.print(f"[bold red]{title}[/bold red]")
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


class ProgressDisplay:
    """
    Context manager for displaying pipeline progress.
    """

    def __init__(self, title: str = "Processing") -> None:
        self._title = title
        self._steps: List[str] = []

    def __enter__(self) -> "ProgressDisplay":
        console.print(f"\n[bold]{self._title}[/bold]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            print_success("Complete")
        else:
            print_error(f"Failed: {exc_val}")

    def step(self, name: str, status: str = "running") -> None:
        """Report a step."""
        if status == "running":
            console.print(f"  [dim]→[/dim] {name}...", end="\r")
        elif status == "done":
            console.print(f"  [green]✓[/green] {name}    ")
        elif status == "skip":
            console.print(f"  [yellow]○[/yellow] {name} (skipped)")
        elif status == "fail":
            console.print(f"  [red]✗[/red] {name} (failed)")

    def update(self, message: str) -> None:
        """Update progress message."""
        console.print(f"  [dim]→[/dim] {message}...", end="\r")


# Aliases for compatibility
display_error = print_error
display_success = print_success
display_info = print_info
display_warning = print_warning


def display_index_stats(
    vector_index: Dict[str, Any],
    bm25_index: Dict[str, Any],
) -> None:
    """
    Display index statistics in a clear, user-friendly format.
    
    Args:
        vector_index: Vector index stats
        bm25_index: BM25 index stats
    """
    console.print()
    console.print("[bold blue]📊 Index Statistics[/bold blue]")
    console.print()
    
    # Vector Index Section
    console.print("[bold cyan]Vector Search (Dense Retrieval)[/bold cyan]")
    
    redis_search_available = vector_index.get("redis_search_available", False)
    index_exists = vector_index.get("exists", False)
    doc_count = vector_index.get("document_count", 0)
    imports_available = vector_index.get("imports_available", True)
    server_has_module = vector_index.get("server_has_module", False)
    
    if redis_search_available:
        if index_exists:
            console.print("  [green]✓[/green] Redis Search: [green]Active[/green] (HNSW index)")
            console.print(f"    Documents indexed: {doc_count:,}")
            if "dimension" in vector_index:
                console.print(f"    Embedding dimension: {vector_index['dimension']}")
        else:
            console.print("  [yellow]![/yellow] Redis Search: Available but index not created")
            console.print("    Run ingestion to create the vector index")
    elif not imports_available and server_has_module:
        # Server has Redis Search but Python imports failed
        console.print("  [red]✗[/red] Redis Search: [red]Import Error[/red]")
        console.print("    Server has Redis Search module, but Python can't use it")
        console.print("    Using fallback: Linear scan (slower)")
        console.print(f"    Documents in store: {doc_count:,}")
        console.print()
        console.print("  [yellow]💡 Fix: Upgrade redis-py package:[/yellow]")
        console.print("     pip install --upgrade redis")
    else:
        console.print("  [yellow]![/yellow] Redis Search: [yellow]Not available[/yellow]")
        console.print("    Using fallback: Linear scan (slower for large datasets)")
        console.print(f"    Documents in store: {doc_count:,}")
        console.print()
        console.print("  [dim]💡 For faster vector search, install Redis Stack:[/dim]")
        console.print("  [dim]   docker run -d -p 6379:6379 redis/redis-stack[/dim]")
    
    console.print()
    
    # BM25 Index Section
    console.print("[bold cyan]BM25 Search (Sparse Retrieval)[/bold cyan]")
    
    bm25_doc_count = bm25_index.get("document_count", 0)
    unique_terms = bm25_index.get("unique_terms", 0)
    needs_rebuild = bm25_index.get("needs_rebuild", False)
    
    if bm25_doc_count > 0:
        console.print("  [green]✓[/green] BM25 Index: [green]Active[/green]")
        console.print(f"    Documents: {bm25_doc_count:,}")
        console.print(f"    Vocabulary: {unique_terms:,} unique terms")
        if needs_rebuild:
            console.print("  [yellow]![/yellow] Index needs sync (run query to auto-sync)")
    else:
        console.print("  [yellow]![/yellow] BM25 Index: Empty")
        console.print("    Run ingestion to build the index")
    
    console.print()


def display_pipeline_result(
    result: Any,
    show_answer: bool = True,
    show_metrics: bool = True,
    show_pipeline_details: bool = True,
) -> None:
    """
    Display a pipeline result with answer, source documents, and metrics.
    
    Args:
        result: PipelineResult object
        show_answer: Whether to display the answer
        show_metrics: Whether to display step metrics
        show_pipeline_details: Whether to show query transformations
    """
    # Display pipeline decisions/transformations first
    if show_pipeline_details and hasattr(result, 'context'):
        display_pipeline_details(result.context)
    
    # Display answer with source documents
    if show_answer:
        # Get scored docs from context (prefer reranked, then fused, etc.)
        scored_docs = None
        if hasattr(result, 'context'):
            ctx = result.context
            # These are List[Tuple[StoredDoc, float]] - already have scores
            scored_docs = (
                getattr(ctx, 'reranked', None) or
                getattr(ctx, 'auto_merged', None) or
                getattr(ctx, 'fused', None) or
                getattr(ctx, 'dense_retrieved', None) or
                getattr(ctx, 'bm25_retrieved', None)
            )
            # Limit to top 8 for display
            if scored_docs:
                scored_docs = scored_docs[:8]
        
        display_answer(
            result.context.original_query,
            result.answer,
            source_docs=scored_docs,
            critic_notes=result.context.critic_notes[0] if result.context.critic_notes else None,
        )
    
    # Display timing metrics if available
    if show_metrics and hasattr(result, 'metrics') and result.metrics and result.metrics.steps:
        console.print()
        display_step_metrics(result.metrics)


def display_pipeline_details(ctx: Any) -> None:
    """
    Display detailed pipeline decisions and query transformations.
    
    Args:
        ctx: AgentContext with pipeline results
    """
    console.print()
    console.print("[bold blue]🔄 Pipeline Details[/bold blue]")
    console.print()
    
    # Planning decisions
    if hasattr(ctx, 'plan') and ctx.plan:
        plan = ctx.plan
        if plan:
            console.print("[cyan]Planning:[/cyan]")
            flags = []
            if plan.get("use_decomposition"):
                flags.append("[green]✓[/green] Decomposition")
            if plan.get("use_rewrite"):
                flags.append("[green]✓[/green] Rewrite")
            if plan.get("use_expansion"):
                flags.append("[green]✓[/green] Expansion")
            if plan.get("complexity"):
                flags.append(f"Complexity: {plan['complexity']}")
            if flags:
                console.print(f"  {' │ '.join(flags)}")
            console.print()
    
    # Query decomposition
    if hasattr(ctx, 'decomposed_queries') and ctx.decomposed_queries and len(ctx.decomposed_queries) > 1:
        console.print("[cyan]Query Decomposition:[/cyan]")
        for i, subq in enumerate(ctx.decomposed_queries, 1):
            console.print(f"  {i}. {subq}")
        console.print()
    
    # Query rewrites
    if hasattr(ctx, 'rewrites') and ctx.rewrites:
        console.print("[cyan]Query Rewrites:[/cyan]")
        for orig, rewritten in ctx.rewrites:
            console.print(f"  [dim]{orig}[/dim]")
            console.print(f"  → {rewritten}")
        console.print()
    
    # Query expansions
    if hasattr(ctx, 'expansions') and ctx.expansions:
        console.print("[cyan]Query Expansions:[/cyan]")
        # Show as comma-separated terms
        terms = ctx.expansions[:12]  # Limit display
        console.print(f"  {', '.join(terms)}")
        if len(ctx.expansions) > 12:
            console.print(f"  [dim]... and {len(ctx.expansions) - 12} more[/dim]")
        console.print()
    
    # Retrieval stats
    dense_count = len(ctx.dense_retrieved) if hasattr(ctx, 'dense_retrieved') and ctx.dense_retrieved else 0
    bm25_count = len(ctx.bm25_retrieved) if hasattr(ctx, 'bm25_retrieved') and ctx.bm25_retrieved else 0
    fused_count = len(ctx.fused) if hasattr(ctx, 'fused') and ctx.fused else 0
    merged_count = len(ctx.auto_merged) if hasattr(ctx, 'auto_merged') and ctx.auto_merged else 0
    reranked_count = len(ctx.reranked) if hasattr(ctx, 'reranked') and ctx.reranked else 0
    
    if dense_count or bm25_count:
        console.print("[cyan]Retrieval:[/cyan]")
        retrieval_info = []
        if dense_count:
            retrieval_info.append(f"Dense: {dense_count}")
        if bm25_count:
            retrieval_info.append(f"BM25: {bm25_count}")
        if fused_count:
            retrieval_info.append(f"Fused: {fused_count}")
        if merged_count:
            retrieval_info.append(f"Merged: {merged_count}")
        if reranked_count:
            retrieval_info.append(f"Reranked: {reranked_count}")
        console.print(f"  {' → '.join(retrieval_info)}")
        console.print()
    
    # Warnings
    if hasattr(ctx, 'warnings') and ctx.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in ctx.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")
        console.print()
