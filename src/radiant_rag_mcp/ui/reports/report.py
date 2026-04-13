"""
Professional report generation for Radiant Agentic RAG.

Provides formatted output for console display and exportable reports
in Markdown and HTML formats.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from radiant_rag_mcp.utils.metrics import RunMetrics
from radiant_rag_mcp.storage.base import StoredDoc

logger = logging.getLogger(__name__)

# Console instance
console = Console()


@dataclass
class QueryReport:
    """Container for query results that can be displayed or exported."""
    
    query: str
    answer: str
    sources: List[Tuple[StoredDoc, float]] = field(default_factory=list)
    
    # Pipeline details
    retrieval_mode: str = "hybrid"
    decomposed_queries: List[str] = field(default_factory=list)
    rewrites: List[Tuple[str, str]] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    
    # Retrieval stats
    dense_count: int = 0
    bm25_count: int = 0
    fused_count: int = 0
    merged_count: int = 0
    reranked_count: int = 0
    
    # Critique
    critique: Optional[Dict[str, Any]] = None
    
    # Metrics
    metrics: Optional[RunMetrics] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = ""
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def from_pipeline_result(cls, result: Any, retrieval_mode: str = "hybrid") -> "QueryReport":
        """Create report from PipelineResult."""
        ctx = result.context
        
        # Get sources with scores
        sources = (
            getattr(ctx, 'reranked', None) or
            getattr(ctx, 'auto_merged', None) or
            getattr(ctx, 'fused', None) or
            getattr(ctx, 'dense_retrieved', None) or
            getattr(ctx, 'bm25_retrieved', None) or
            []
        )
        
        return cls(
            query=ctx.original_query,
            answer=result.answer,
            sources=sources[:10],  # Top 10 for report
            retrieval_mode=retrieval_mode,
            decomposed_queries=ctx.decomposed_queries or [],
            rewrites=ctx.rewrites or [],
            expansions=ctx.expansions or [],
            plan=ctx.plan or {},
            dense_count=len(ctx.dense_retrieved) if ctx.dense_retrieved else 0,
            bm25_count=len(ctx.bm25_retrieved) if ctx.bm25_retrieved else 0,
            fused_count=len(ctx.fused) if ctx.fused else 0,
            merged_count=len(ctx.auto_merged) if ctx.auto_merged else 0,
            reranked_count=len(ctx.reranked) if ctx.reranked else 0,
            critique=ctx.critic_notes[0] if ctx.critic_notes else None,
            metrics=result.metrics,
            run_id=ctx.run_id,
            warnings=ctx.warnings or [],
        )


def normalize_scores(sources: List[Tuple[StoredDoc, float]]) -> List[Tuple[StoredDoc, float, float]]:
    """
    Normalize relevance scores to 0-100 range for display.
    
    Cross-encoder and other retrieval scores can be logits that vary in range
    and may be negative. This function applies min-max normalization to convert
    scores to a consistent 0-100 percentage scale for user-friendly display.
    
    The highest score in the result set becomes 100%, and the lowest becomes 0%.
    This relative normalization helps users understand document relevance within
    the context of a specific query, regardless of the underlying scoring method.
    
    Args:
        sources: List of (StoredDoc, score) tuples from retrieval.
        
    Returns:
        List of (StoredDoc, raw_score, normalized_score) tuples where:
        - StoredDoc: The original document object
        - raw_score: The original retrieval score
        - normalized_score: Score normalized to 0-100 range
        
    Edge Cases:
        - Empty list returns empty list
        - Single document gets 100%
        - All identical scores get 100% each
        
    Examples:
        >>> docs_scores = [(doc1, 0.9), (doc2, 0.5), (doc3, 0.1)]
        >>> normalized = normalize_scores(docs_scores)
        >>> # Returns [(doc1, 0.9, 100.0), (doc2, 0.5, 50.0), (doc3, 0.1, 0.0)]
    """
    if not sources:
        return []
    
    if len(sources) == 1:
        # Single result gets 100%
        return [(sources[0][0], sources[0][1], 100.0)]
    
    scores = [score for _, score in sources]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range == 0:
        # All same score - give them all 100%
        return [(doc, score, 100.0) for doc, score in sources]
    
    result = []
    for doc, score in sources:
        # Min-max normalize to 0-100
        # Highest score = 100%, lowest = 0%
        normalized = ((score - min_score) / score_range) * 100
        result.append((doc, score, normalized))
    
    return result


def display_report(report: QueryReport, show_metrics: bool = True, compact: bool = False) -> None:
    """
    Display a professional query report to console.
    
    Args:
        report: QueryReport to display
        show_metrics: Whether to show timing metrics
        compact: Use compact display (less whitespace)
    """
    console.print()
    
    # Header bar
    mode_labels = {
        "hybrid": "Hybrid Search",
        "dense": "Semantic Search", 
        "bm25": "Keyword Search"
    }
    mode_label = mode_labels.get(report.retrieval_mode, report.retrieval_mode)
    
    # Professional header
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        "[bold blue]RADIANT[/bold blue] [dim]Agentic RAG[/dim]",
        f"[dim]{mode_label} • {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]"
    )
    console.print(Panel(header, style="blue", box=box.HEAVY))
    
    # Query section
    console.print()
    console.print("[bold]Query[/bold]")
    console.print(f"  {report.query}")
    
    # Pipeline summary (single line, compact)
    if report.decomposed_queries or report.expansions or report.rewrites:
        console.print()
        pipeline_info = []
        
        if len(report.decomposed_queries) > 1:
            pipeline_info.append(f"[cyan]Decomposed:[/cyan] {len(report.decomposed_queries)} queries")
        
        if report.expansions:
            exp_preview = ", ".join(report.expansions[:4])
            if len(report.expansions) > 4:
                exp_preview += f" [dim](+{len(report.expansions) - 4})[/dim]"
            pipeline_info.append(f"[cyan]Expanded:[/cyan] {exp_preview}")
        
        if report.rewrites:
            pipeline_info.append(f"[cyan]Rewritten:[/cyan] {len(report.rewrites)}")
        
        if pipeline_info:
            console.print(f"[dim]Pipeline:[/dim] {' │ '.join(pipeline_info)}")
        
        # Retrieval flow (compact)
        flow_parts = []
        if report.dense_count:
            flow_parts.append(f"Dense({report.dense_count})")
        if report.bm25_count:
            flow_parts.append(f"BM25({report.bm25_count})")
        if report.fused_count and report.retrieval_mode == "hybrid":
            flow_parts.append(f"Fused({report.fused_count})")
        if report.reranked_count:
            flow_parts.append(f"Reranked({report.reranked_count})")
        
        if flow_parts:
            console.print(f"[dim]Retrieval:[/dim] {' → '.join(flow_parts)}")
    
    console.print()
    
    # Answer - the main content
    console.print(Panel(
        report.answer,
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
        box=box.ROUNDED,
    ))
    
    # Sources table - professional styling
    if report.sources:
        console.print()
        normalized = normalize_scores(report.sources)
        
        # Create table with professional styling
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Source", style="cyan", no_wrap=True, max_width=30)
        table.add_column("Pg", justify="center", width=4)
        table.add_column("Relevance", justify="left", width=14)
        table.add_column("Content", style="dim", max_width=45, overflow="ellipsis")
        
        for i, (doc, raw_score, norm_score) in enumerate(normalized[:6], 1):
            # Source name
            source = doc.meta.get("source_path", doc.meta.get("source", "unknown"))
            if "/" in source:
                source = source.split("/")[-1]
            if len(source) > 30:
                source = source[:27] + "..."
            
            # Page
            page = str(doc.meta.get("page_number", "-"))
            
            # Relevance bar (visual)
            bar_filled = int(norm_score / 10)
            bar_empty = 10 - bar_filled
            if norm_score >= 70:
                bar = f"[green]{'━' * bar_filled}[/green][dim]{'─' * bar_empty}[/dim] {norm_score:.0f}%"
            elif norm_score >= 40:
                bar = f"[yellow]{'━' * bar_filled}[/yellow][dim]{'─' * bar_empty}[/dim] {norm_score:.0f}%"
            else:
                bar = f"[red]{'━' * bar_filled}[/red][dim]{'─' * bar_empty}[/dim] {norm_score:.0f}%"
            
            # Content preview
            preview = doc.content[:90].replace("\n", " ").strip()
            if len(doc.content) > 90:
                preview += "…"
            
            table.add_row(str(i), source, page, bar, preview)
        
        console.print("[bold]Sources[/bold]")
        console.print(table)
        
        if len(normalized) > 6:
            console.print(f"  [dim]... and {len(normalized) - 6} more sources[/dim]")
    
    # Quality assessment (compact, inline)
    if report.critique:
        console.print()
        critique = report.critique
        
        # Scores inline
        scores_parts = []
        for name, key in [("Relevance", "relevance_score"), 
                          ("Faithfulness", "faithfulness_score"),
                          ("Coverage", "coverage_score")]:
            if key in critique:
                score = critique[key]
                if score >= 7:
                    scores_parts.append(f"{name}: [green]{score}/10[/green]")
                elif score >= 5:
                    scores_parts.append(f"{name}: [yellow]{score}/10[/yellow]")
                else:
                    scores_parts.append(f"{name}: [red]{score}/10[/red]")
        
        if scores_parts:
            console.print(f"[bold]Quality[/bold]  {' │ '.join(scores_parts)}")
        
        # Issues (brief)
        issues = critique.get("issues", [])
        if issues and not compact:
            console.print(f"[dim]  Issues: {len(issues)} identified[/dim]")
    
    # Performance (single line)
    if show_metrics and report.metrics:
        total_ms = report.metrics.total_latency_ms
        if total_ms:
            console.print()
            console.print(f"[dim]Completed in {total_ms/1000:.2f}s[/dim]")
    
    # Warnings
    if report.warnings:
        console.print()
        for warning in report.warnings[:2]:
            console.print(f"[yellow]⚠ {warning}[/yellow]")
    
    console.print()


def generate_markdown_report(report: QueryReport) -> str:
    """Generate a Markdown report."""
    lines = []
    
    # Header
    lines.append("# Radiant RAG Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Mode:** {report.retrieval_mode.title()}")
    if report.run_id:
        lines.append(f"**Run ID:** `{report.run_id}`")
    lines.append("")
    
    # Query
    lines.append("## Query")
    lines.append("")
    lines.append(f"> {report.query}")
    lines.append("")
    
    # Pipeline Details
    if report.decomposed_queries or report.expansions or report.rewrites:
        lines.append("## Pipeline Details")
        lines.append("")
        
        if len(report.decomposed_queries) > 1:
            lines.append("**Query Decomposition:**")
            for i, q in enumerate(report.decomposed_queries, 1):
                lines.append(f"{i}. {q}")
            lines.append("")
        
        if report.rewrites:
            lines.append("**Query Rewrites:**")
            for orig, rewritten in report.rewrites:
                lines.append(f"- `{orig}` → `{rewritten}`")
            lines.append("")
        
        if report.expansions:
            lines.append(f"**Query Expansions:** {', '.join(report.expansions)}")
            lines.append("")
        
        # Retrieval stats
        lines.append("**Retrieval Flow:**")
        flow = []
        if report.dense_count:
            flow.append(f"Dense: {report.dense_count}")
        if report.bm25_count:
            flow.append(f"BM25: {report.bm25_count}")
        if report.fused_count:
            flow.append(f"Fused: {report.fused_count}")
        if report.merged_count:
            flow.append(f"Merged: {report.merged_count}")
        if report.reranked_count:
            flow.append(f"Reranked: {report.reranked_count}")
        lines.append(" → ".join(flow))
        lines.append("")
    
    # Answer
    lines.append("## Answer")
    lines.append("")
    lines.append(report.answer)
    lines.append("")
    
    # Sources
    if report.sources:
        lines.append("## Sources")
        lines.append("")
        lines.append("| # | Document | Page | Relevance | Preview |")
        lines.append("|---|----------|------|-----------|---------|")
        
        normalized = normalize_scores(report.sources)
        for i, (doc, raw_score, norm_score) in enumerate(normalized[:10], 1):
            source = doc.meta.get("source_path", "unknown")
            if "/" in source:
                source = source.split("/")[-1]
            page = str(doc.meta.get("page_number", "-"))
            preview = doc.content[:80].replace("\n", " ").replace("|", "\\|").strip()
            if len(doc.content) > 80:
                preview += "..."
            
            relevance = f"{norm_score:.0f}%"
            lines.append(f"| {i} | {source} | {page} | {relevance} | {preview} |")
        
        lines.append("")
    
    # Critique
    if report.critique:
        lines.append("## Quality Assessment")
        lines.append("")
        
        scores = []
        for name, key in [("Relevance", "relevance_score"),
                          ("Faithfulness", "faithfulness_score"),
                          ("Coverage", "coverage_score")]:
            if key in report.critique:
                scores.append(f"**{name}:** {report.critique[key]}/10")
        
        if scores:
            lines.append(" | ".join(scores))
            lines.append("")
        
        issues = report.critique.get("issues", [])
        if issues:
            lines.append("**Issues:**")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        suggestions = report.critique.get("suggested_improvements", [])
        if suggestions:
            lines.append("**Suggestions:**")
            for suggestion in suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")
    
    # Metrics
    if report.metrics and report.metrics.steps:
        lines.append("## Performance")
        lines.append("")
        lines.append("| Step | Latency | Details |")
        lines.append("|------|---------|---------|")
        
        for step in report.metrics.steps:
            latency = f"{step.latency_ms:.0f}ms" if step.latency_ms else "-"
            details = ", ".join(f"{k}={v}" for k, v in step.extra.items() 
                               if isinstance(v, (int, float, str)) and len(str(v)) < 30)
            lines.append(f"| {step.name} | {latency} | {details} |")
        
        lines.append("")
        if report.metrics.total_latency_ms:
            lines.append(f"**Total Time:** {report.metrics.total_latency_ms/1000:.2f}s")
            lines.append("")
    
    # Warnings
    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    return "\n".join(lines)


def generate_html_report(report: QueryReport) -> str:
    """Generate an HTML report with styling."""
    md_content = generate_markdown_report(report)
    
    # Convert markdown to basic HTML
    html_content = _markdown_to_html(md_content)
    
    template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radiant RAG Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #ca8a04;
            --error: #dc2626;
            --bg: #ffffff;
            --bg-alt: #f8fafc;
            --text: #1e293b;
            --text-dim: #64748b;
            --border: #e2e8f0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text);
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg);
        }}
        
        h1 {{
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
        }}
        
        h2 {{
            color: var(--text);
            margin-top: 2rem;
            font-size: 1.25rem;
        }}
        
        blockquote {{
            background: var(--bg-alt);
            border-left: 4px solid var(--primary);
            margin: 1rem 0;
            padding: 1rem;
            font-style: italic;
        }}
        
        .answer {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid var(--success);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: var(--bg-alt);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: var(--bg-alt);
        }}
        
        code {{
            background: var(--bg-alt);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .meta {{
            color: var(--text-dim);
            font-size: 0.9rem;
        }}
        
        .score {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }}
        
        .score-high {{ background: #dcfce7; color: var(--success); }}
        .score-med {{ background: #fef9c3; color: var(--warning); }}
        .score-low {{ background: #fee2e2; color: var(--error); }}
        
        ul {{
            padding-left: 1.5rem;
        }}
        
        li {{
            margin: 0.25rem 0;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 2rem 0;
        }}
        
        .footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-dim);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    {html_content}
    <div class="footer">
        <p>Generated by Radiant Agentic RAG • {report.timestamp}</p>
    </div>
</body>
</html>'''
    
    return template


def _markdown_to_html(md: str) -> str:
    """Simple markdown to HTML conversion."""
    import re
    
    lines = md.split("\n")
    html_lines = []
    in_table = False
    in_list = False
    table_row_count = 0
    
    for line in lines:
        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        # Blockquote
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        # Table
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                table_row_count = 0
            
            if line.startswith("|---"):
                continue  # Skip separator
            
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # First row is header
            tag = "th" if table_row_count == 0 else "td"
            row = "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
            table_row_count += 1
        # List item
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(line[2:])}</li>")
        # Numbered list
        elif line and line[0].isdigit() and ". " in line[:4]:
            html_lines.append(f"<p>{html.escape(line)}</p>")
        # Empty line
        elif not line.strip():
            if in_table:
                html_lines.append("</table>")
                in_table = False
                table_row_count = 0
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
        # Bold
        elif "**" in line:
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f"<p>{line}</p>")
        # Code
        elif "`" in line:
            line = html.escape(line)
            line = re.sub(r'`(.+?)`', r'<code>\1</code>', line)
            html_lines.append(f"<p>{line}</p>")
        # Regular paragraph
        else:
            html_lines.append(f"<p>{html.escape(line)}</p>")
    
    # Close any open tags
    if in_table:
        html_lines.append("</table>")
    if in_list:
        html_lines.append("</ul>")
    
    return "\n".join(html_lines)


def save_report(
    report: QueryReport,
    filepath: str,
    format: str = "auto",
) -> str:
    """
    Save report to file.
    
    Args:
        report: QueryReport to save
        filepath: Output file path
        format: "markdown", "html", "json", or "auto" (detect from extension)
        
    Returns:
        Absolute path to saved file
    """
    path = Path(filepath)
    
    # Auto-detect format from extension
    if format == "auto":
        ext = path.suffix.lower()
        if ext in (".md", ".markdown"):
            format = "markdown"
        elif ext in (".html", ".htm"):
            format = "html"
        elif ext == ".json":
            format = "json"
        else:
            format = "markdown"
            path = path.with_suffix(".md")
    
    # Generate content
    if format == "markdown":
        content = generate_markdown_report(report)
    elif format == "html":
        content = generate_html_report(report)
    elif format == "json":
        # JSON export
        data = {
            "query": report.query,
            "answer": report.answer,
            "retrieval_mode": report.retrieval_mode,
            "timestamp": report.timestamp,
            "run_id": report.run_id,
            "sources": [
                {
                    "content": doc.content[:500],
                    "source": doc.meta.get("source_path", "unknown"),
                    "page": doc.meta.get("page_number"),
                    "score": score,
                }
                for doc, score in report.sources
            ],
            "pipeline": {
                "decomposed_queries": report.decomposed_queries,
                "expansions": report.expansions,
                "rewrites": [(o, r) for o, r in report.rewrites],
            },
            "retrieval_stats": {
                "dense": report.dense_count,
                "bm25": report.bm25_count,
                "fused": report.fused_count,
                "merged": report.merged_count,
                "reranked": report.reranked_count,
            },
            "critique": report.critique,
            "warnings": report.warnings,
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    path.write_text(content, encoding="utf-8")
    
    logger.info(f"Report saved to {path.absolute()}")
    return str(path.absolute())


# Convenience function for quick display
def print_report(
    result: Any,
    retrieval_mode: str = "hybrid",
    show_metrics: bool = True,
    save_path: Optional[str] = None,
    compact: bool = False,
) -> Optional[str]:
    """
    Display pipeline result as a professional report.
    
    Args:
        result: PipelineResult from orchestrator
        retrieval_mode: Retrieval mode used
        show_metrics: Show timing metrics
        save_path: Optional path to save report
        compact: Use compact display
        
    Returns:
        Path to saved report if save_path provided
    """
    report = QueryReport.from_pipeline_result(result, retrieval_mode)
    display_report(report, show_metrics, compact)
    
    if save_path:
        return save_report(report, save_path)
    
    return None


def display_search_results(
    query: str,
    results: List[Tuple[StoredDoc, float]],
    mode: str = "hybrid",
    show_content: bool = True,
) -> None:
    """
    Display search-only results (no LLM generation) in professional format.
    
    Args:
        query: Search query
        results: List of (doc, score) tuples
        mode: Retrieval mode used
        show_content: Whether to show content previews
    """
    console.print()
    
    # Header
    mode_labels = {
        "hybrid": "Hybrid Search",
        "dense": "Semantic Search", 
        "bm25": "Keyword Search"
    }
    mode_label = mode_labels.get(mode, mode)
    
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        "[bold blue]RADIANT[/bold blue] [dim]Document Search[/dim]",
        f"[dim]{mode_label} • {len(results)} results[/dim]"
    )
    console.print(Panel(header, style="blue", box=box.HEAVY))
    
    # Query
    console.print()
    console.print(f"[bold]Query[/bold]  {query}")
    console.print()
    
    if not results:
        console.print("[yellow]No documents found matching your query.[/yellow]")
        console.print()
        return
    
    # Results table
    normalized = normalize_scores(results)
    
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Document", style="cyan", no_wrap=True, max_width=32)
    table.add_column("Page", justify="center", width=5)
    table.add_column("Score", justify="left", width=14)
    
    if show_content:
        table.add_column("Content Preview", style="dim", max_width=50)
    
    for i, (doc, raw_score, norm_score) in enumerate(normalized, 1):
        # Source
        source = doc.meta.get("source_path", doc.meta.get("source", "unknown"))
        if "/" in source:
            source = source.split("/")[-1]
        if len(source) > 32:
            source = source[:29] + "..."
        
        # Page
        page = str(doc.meta.get("page_number", "-"))
        
        # Score bar
        bar_filled = int(norm_score / 10)
        bar_empty = 10 - bar_filled
        if norm_score >= 70:
            score_display = f"[green]{'━' * bar_filled}[/green][dim]{'─' * bar_empty}[/dim] {norm_score:.0f}%"
        elif norm_score >= 40:
            score_display = f"[yellow]{'━' * bar_filled}[/yellow][dim]{'─' * bar_empty}[/dim] {norm_score:.0f}%"
        else:
            score_display = f"[red]{'━' * bar_filled}[/red][dim]{'─' * bar_empty}[/dim] {norm_score:.0f}%"
        
        if show_content:
            preview = doc.content[:100].replace("\n", " ").strip()
            if len(doc.content) > 100:
                preview += "…"
            table.add_row(str(i), source, page, score_display, preview)
        else:
            table.add_row(str(i), source, page, score_display)
    
    console.print(table)
    console.print()


@dataclass  
class SearchReport:
    """Container for search-only results."""
    
    query: str
    results: List[Tuple[StoredDoc, float]]
    mode: str = "hybrid"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Document Search Results",
            "",
            f"**Query:** {self.query}",
            f"**Mode:** {self.mode.title()}",
            f"**Results:** {len(self.results)}",
            f"**Date:** {self.timestamp}",
            "",
            "## Results",
            "",
            "| # | Document | Page | Score | Preview |",
            "|---|----------|------|-------|---------|",
        ]
        
        normalized = normalize_scores(self.results)
        for i, (doc, raw, norm) in enumerate(normalized, 1):
            source = doc.meta.get("source_path", "unknown")
            if "/" in source:
                source = source.split("/")[-1]
            page = str(doc.meta.get("page_number", "-"))
            preview = doc.content[:60].replace("\n", " ").replace("|", "\\|")
            if len(doc.content) > 60:
                preview += "..."
            lines.append(f"| {i} | {source} | {page} | {norm:.0f}% | {preview} |")
        
        return "\n".join(lines)


def save_search_report(
    query: str,
    results: List[Tuple[StoredDoc, float]],
    filepath: str,
    mode: str = "hybrid",
) -> str:
    """Save search results to file."""
    report = SearchReport(query=query, results=results, mode=mode)
    
    path = Path(filepath)
    ext = path.suffix.lower()
    
    if ext == ".json":
        data = {
            "query": query,
            "mode": mode,
            "timestamp": report.timestamp,
            "results": [
                {
                    "source": doc.meta.get("source_path", "unknown"),
                    "page": doc.meta.get("page_number"),
                    "score": score,
                    "content": doc.content[:500],
                }
                for doc, score in results
            ]
        }
        content = json.dumps(data, indent=2)
    else:
        content = report.to_markdown()
        if not path.suffix:
            path = path.with_suffix(".md")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    
    return str(path.absolute())
