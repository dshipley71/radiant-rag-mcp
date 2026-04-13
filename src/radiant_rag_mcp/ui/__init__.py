"""
User interface modules package.

Provides:
    - Console display utilities
    - Textual TUI
    - Report generation (HTML, Markdown, JSON, Text)
"""

from radiant_rag_mcp.ui.display import (
    console,
    display_info,
    display_success,
    display_warning,
    display_error,
    display_index_stats,
    ProgressDisplay,
)

__all__ = [
    # Display
    "console",
    "display_info",
    "display_success",
    "display_warning",
    "display_error",
    "display_index_stats",
    "ProgressDisplay",
]
