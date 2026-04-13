"""
Report generation for RAG system.

Provides various report formats:
    - QueryReport: Base report structure
    - Markdown reports
    - HTML reports
    - JSON reports
    - Text reports
"""

from radiant_rag_mcp.ui.reports.report import (
    QueryReport,
    SearchReport,
    normalize_scores,
    display_report,
    generate_markdown_report,
    generate_html_report,
    save_report,
    print_report,
    display_search_results,
    save_search_report,
)

from radiant_rag_mcp.ui.reports.text import (
    ReportConfig,
    TextReportBuilder,
    generate_text_report,
    save_text_report,
    print_text_report,
)

__all__ = [
    # Query reports
    "QueryReport",
    "SearchReport",
    "normalize_scores",
    "display_report",
    "generate_markdown_report",
    "generate_html_report",
    "save_report",
    "print_report",
    "display_search_results",
    "save_search_report",
    # Text reports
    "ReportConfig",
    "TextReportBuilder",
    "generate_text_report",
    "save_text_report",
    "print_text_report",
]
