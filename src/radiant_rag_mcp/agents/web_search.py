"""
Web search agent for RAG pipeline.

Provides real-time web content augmentation during queries.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import WebSearchConfig
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


class WebSearchAgent(LLMAgent):
    """
    Real-time web search agent for augmenting indexed content.
    
    Fetches and processes web content during query time to provide
    current information that may not be in the indexed corpus.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "WebSearchConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the web search agent.
        
        Args:
            llm: LLM client for query analysis
            config: Web search configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)
        self._config = config
        self._cache: Dict[str, List[Tuple[Any, float]]] = {}

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "WebSearchAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.RETRIEVAL

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Real-time web search agent for augmenting indexed content"

    def _should_search(self, query: str, plan: Dict[str, Any]) -> bool:
        """
        Determine if web search should be performed.
        """
        if plan.get("use_web_search", False):
            return True
        
        query_lower = query.lower()
        for keyword in self._config.trigger_keywords:
            if keyword.lower() in query_lower:
                return True
        
        return False

    def _analyze_query_for_urls(self, query: str) -> List[str]:
        """
        Use LLM to suggest URLs that might contain relevant information.
        """
        system = """You are a search assistant. Given a query, suggest 1-3 specific URLs
that are likely to contain authoritative, relevant information.

CRITICAL: Only suggest URLs you are CERTAIN exist. Use well-known, stable domains.
Do NOT invent URLs - if uncertain about the exact path, suggest the domain homepage.

Reliable sources:
- Wikipedia: https://en.wikipedia.org/wiki/[Topic] (for factual queries)
- ArXiv: https://arxiv.org/ (for research papers)
- GitHub: https://github.com/ (for code/technical topics)
- Official documentation sites (Python.org, Mozilla MDN, etc.)
- News sites: Reuters, AP News, BBC (for current events)
- Government/academic sites (.gov, .edu domains)

AVOID inventing specific article URLs. When in doubt, use homepage URLs like:
- https://en.wikipedia.org/ (then user can search)
- https://stackoverflow.com/
- https://github.com/"""

        # Add preferred domains guidance if configured
        if self._config.preferred_domains:
            domains_str = ", ".join(self._config.preferred_domains)
            system += f"""

Prioritize these domains when relevant: {domains_str}
However, still suggest other authoritative sources if they're clearly more relevant to the query."""

        system += """

Return ONLY raw JSON array with no markdown formatting. Do not wrap in ```json code blocks.
Example: ["https://en.wikipedia.org/wiki/Topic", "https://docs.example.com/guide"]

If no good URLs come to mind, return an empty array: []"""

        user = f"Query: {query}\n\nReturn raw JSON array only, no code blocks."

        result = self._chat_json(
            system=system,
            user=user,
            default=[],
            expected_type=list,
        )

        if not result:
            self.logger.warning("LLM returned empty array for URL suggestions")
            return []

        self.logger.info(f"LLM suggested {len(result)} URLs: {result}")

        valid_urls = []
        for url in result[:self._config.max_results]:
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                blocked = False
                for domain in self._config.blocked_domains:
                    if domain.lower() in url.lower():
                        blocked = True
                        self.logger.debug(f"Blocked URL (matches blocked domain '{domain}'): {url}")
                        break
                if not blocked:
                    valid_urls.append(url)
            else:
                self.logger.debug(f"Invalid URL format: {url}")

        self.logger.info(f"Validated {len(valid_urls)} URLs from {len(result)} suggestions")
        return valid_urls

    def _fetch_and_parse(self, urls: List[str]) -> List[Tuple[Any, float]]:
        """
        Fetch URLs and convert to StoredDoc format.
        """
        from radiant_rag_mcp.ingestion.web_crawler import WebCrawler
        from radiant_rag_mcp.storage.base import StoredDoc
        
        results: List[Tuple[StoredDoc, float]] = []
        
        crawler = WebCrawler(
            max_depth=0,
            max_pages=self._config.max_pages,
            timeout=self._config.timeout,
            user_agent=self._config.user_agent,
        )
        
        try:
            for i, url in enumerate(urls[:self._config.max_pages]):
                self.logger.debug(f"Fetching URL {i+1}/{len(urls[:self._config.max_pages])}: {url}")
                try:
                    crawl_result = crawler.crawl_single(url, save_file=False)

                    if not crawl_result.success or not crawl_result.content:
                        self.logger.warning(f"Failed to fetch {url}: {crawl_result.error}")
                        continue
                    
                    if crawl_result.is_html:
                        content = self._extract_text_from_html(
                            crawl_result.content.decode("utf-8", errors="replace")
                        )
                    else:
                        content = crawl_result.content.decode("utf-8", errors="replace")
                    
                    max_content = 10000
                    if len(content) > max_content:
                        content = content[:max_content] + "..."
                    
                    if not content.strip():
                        continue
                    
                    doc_id = f"web_search_{hash(url)}"
                    doc = StoredDoc(
                        doc_id=doc_id,
                        content=content,
                        meta={
                            "source_type": "web_search",
                            "source_url": url,
                            "page_title": crawl_result.title or url,
                            "content_type": crawl_result.content_type,
                            "fetched_at": str(int(time.time())),
                        },
                    )
                    
                    relevance = 1.0 - (i * 0.1)
                    relevance = max(relevance, self._config.min_relevance)
                    
                    results.append((doc, relevance))
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching {url}: {e}")
                    continue
                    
        finally:
            crawler.close()
        
        return results

    def _extract_text_from_html(self, html: str) -> str:
        """
        Extract readable text from HTML content.
        """
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", html)
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _execute(
        self,
        query: str,
        plan: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Execute web search for the query.
        
        Args:
            query: User query
            plan: Execution plan
            
        Returns:
            List of (document, relevance) tuples
        """
        if not self._config.enabled:
            return []
        
        if not self._should_search(query, plan):
            self.logger.debug(f"Web search not triggered for query: {query[:50]}...")
            return []
        
        cache_key = query.lower().strip()
        if self._config.cache_enabled and cache_key in self._cache:
            self.logger.debug(f"Returning cached web search results for: {query[:50]}...")
            return self._cache[cache_key]
        
        self.logger.info(f"Performing web search for: {query[:50]}...")
        
        urls = self._analyze_query_for_urls(query)
        
        if not urls:
            self.logger.debug("No URLs suggested for query")
            return []
        
        self.logger.debug(f"Fetching {len(urls)} URLs: {urls}")
        
        results = self._fetch_and_parse(urls)
        
        if self._config.cache_enabled:
            self._cache[cache_key] = results
        
        self.logger.info(f"Web search returned {len(results)} results")
        
        return results

    def clear_cache(self) -> None:
        """Clear the web search cache."""
        self._cache.clear()

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[List[Tuple[Any, float]]]:
        """
        Return empty list on error.
        """
        self.logger.warning(f"Web search failed: {error}")
        return []
