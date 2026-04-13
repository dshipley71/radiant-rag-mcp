"""
radiant-rag-mcp — Radiant Agentic RAG as an MCP server.

Exposes the full Radiant RAG pipeline as ten MCP tools consumable by
Claude Code, Claude Desktop, and any MCP-compatible LLM client.
"""

__version__ = "2.0.0"
__author__ = "Radiant RAG Team"

# Defer imports to avoid circular dependencies and blocking I/O at import time.
# Use: from radiant_rag_mcp.app import RadiantRAG, create_app
# Use: from radiant_rag_mcp.config import load_config

__all__ = ["__version__"]
