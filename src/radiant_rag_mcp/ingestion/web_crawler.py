"""
Web crawler for URL-based document ingestion.

Provides URL fetching, link extraction, and crawling with configurable
depth limits. Supports multiple content types and basic authentication.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from requests.auth import HTTPBasicAuth

if TYPE_CHECKING:
    from radiant_rag_mcp.config import WebCrawlerConfig

logger = logging.getLogger(__name__)

# Content types we can process
SUPPORTED_CONTENT_TYPES = {
    # HTML
    "text/html",
    "application/xhtml+xml",
    # Plain text
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    # PDF
    "application/pdf",
    # Documents
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # JSON/XML
    "application/json",
    "application/xml",
    "text/xml",
    # Rich text
    "application/rtf",
}

# File extensions for content type detection
EXTENSION_TO_CONTENT_TYPE = {
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".json": "application/json",
    ".xml": "application/xml",
    ".rtf": "application/rtf",
}


@dataclass
class CrawlResult:
    """Result from crawling a single URL."""

    url: str
    content_type: str
    content: bytes
    local_path: Optional[str] = None
    title: Optional[str] = None
    links: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True

    @property
    def is_html(self) -> bool:
        """Check if content is HTML."""
        return self.content_type in ("text/html", "application/xhtml+xml")


@dataclass
class CrawlStats:
    """Statistics from a crawl operation."""

    urls_discovered: int = 0
    urls_crawled: int = 0
    urls_skipped: int = 0
    urls_failed: int = 0
    bytes_downloaded: int = 0
    pages_by_depth: Dict[int, int] = field(default_factory=dict)
    content_types: Dict[str, int] = field(default_factory=dict)
    errors: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urls_discovered": self.urls_discovered,
            "urls_crawled": self.urls_crawled,
            "urls_skipped": self.urls_skipped,
            "urls_failed": self.urls_failed,
            "bytes_downloaded": self.bytes_downloaded,
            "pages_by_depth": self.pages_by_depth,
            "content_types": self.content_types,
            "errors": self.errors[:10],  # Limit errors in output
        }


class URLNormalizer:
    """Normalize URLs for consistent comparison and deduplication."""

    @staticmethod
    def normalize(url: str) -> str:
        """
        Normalize a URL for comparison.

        - Converts to lowercase scheme and host
        - Removes default ports
        - Removes trailing slashes (except for root)
        - Removes fragments
        - Sorts query parameters
        """
        parsed = urlparse(url)

        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Normalize path
        path = parsed.path
        if path and path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        if not path:
            path = "/"

        # Sort query parameters
        query = parsed.query
        if query:
            params = sorted(query.split("&"))
            query = "&".join(params)

        # Reconstruct without fragment
        return urlunparse((scheme, netloc, path, parsed.params, query, ""))

    @staticmethod
    def get_domain(url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    @staticmethod
    def is_same_domain(url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        return URLNormalizer.get_domain(url1) == URLNormalizer.get_domain(url2)


class LinkExtractor:
    """Extract links from HTML content."""

    # Regex patterns for link extraction
    HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
    SRC_PATTERN = re.compile(r'src=["\']([^"\']+)["\']', re.IGNORECASE)
    TITLE_PATTERN = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)

    @classmethod
    def extract_links(cls, html: str, base_url: str) -> List[str]:
        """
        Extract all links from HTML content.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        links = set()

        # Extract href attributes
        for match in cls.HREF_PATTERN.finditer(html):
            href = match.group(1).strip()
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                absolute_url = urljoin(base_url, href)
                links.add(absolute_url)

        return list(links)

    @classmethod
    def extract_title(cls, html: str) -> Optional[str]:
        """Extract page title from HTML."""
        match = cls.TITLE_PATTERN.search(html)
        if match:
            return match.group(1).strip()
        return None


class WebCrawler:
    """
    Web crawler for URL-based document ingestion.

    Features:
    - Configurable crawl depth
    - Same-domain restriction option
    - URL filtering with include/exclude patterns
    - Basic authentication support
    - Rate limiting
    - Content type filtering
    """

    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 100,
        same_domain_only: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        timeout: int = 30,
        delay: float = 0.5,
        user_agent: str = "AgenticRAG-Crawler/1.0",
        basic_auth: Optional[Tuple[str, str]] = None,
        verify_ssl: bool = True,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize the web crawler.

        Args:
            max_depth: Maximum crawl depth (0 = seed URLs only)
            max_pages: Maximum total pages to crawl
            same_domain_only: Only crawl pages from same domain as seed
            include_patterns: Regex patterns - URLs must match at least one
            exclude_patterns: Regex patterns - URLs matching any are skipped
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
            user_agent: User agent string
            basic_auth: Optional (username, password) for basic auth
            verify_ssl: Whether to verify SSL certificates
            temp_dir: Directory for temporary files
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        self.timeout = timeout
        self.delay = delay
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl
        self.temp_dir = temp_dir

        # Authentication
        self._auth = HTTPBasicAuth(*basic_auth) if basic_auth else None

        # URL patterns
        self._include_patterns = [
            re.compile(p) for p in (include_patterns or [])
        ]
        self._exclude_patterns = [
            re.compile(p) for p in (exclude_patterns or [])
        ]

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

        # Tracking
        self._visited: Set[str] = set()
        self._normalizer = URLNormalizer()

    @classmethod
    def from_config(cls, config: "WebCrawlerConfig") -> "WebCrawler":
        """Create crawler from configuration."""
        basic_auth = None
        if config.basic_auth_user and config.basic_auth_password:
            basic_auth = (config.basic_auth_user, config.basic_auth_password)

        return cls(
            max_depth=config.max_depth,
            max_pages=config.max_pages,
            same_domain_only=config.same_domain_only,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
            timeout=config.timeout,
            delay=config.delay,
            user_agent=config.user_agent,
            basic_auth=basic_auth,
            verify_ssl=config.verify_ssl,
            temp_dir=config.temp_dir,
        )

    def _should_crawl(self, url: str, seed_domain: str) -> bool:
        """Check if URL should be crawled based on filters."""
        # Normalize URL
        normalized = self._normalizer.normalize(url)

        # Already visited?
        if normalized in self._visited:
            return False

        # Check scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        # Same domain check
        if self.same_domain_only:
            if self._normalizer.get_domain(url) != seed_domain:
                return False

        # Include patterns (must match at least one if specified)
        if self._include_patterns:
            if not any(p.search(url) for p in self._include_patterns):
                return False

        # Exclude patterns (must not match any)
        if self._exclude_patterns:
            if any(p.search(url) for p in self._exclude_patterns):
                return False

        return True

    def _get_content_type(self, response: requests.Response, url: str) -> str:
        """Determine content type from response or URL."""
        # Try Content-Type header
        content_type = response.headers.get("Content-Type", "")
        if content_type:
            # Strip charset and other parameters
            content_type = content_type.split(";")[0].strip().lower()
            if content_type in SUPPORTED_CONTENT_TYPES:
                return content_type

        # Fall back to URL extension
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        if ext in EXTENSION_TO_CONTENT_TYPE:
            return EXTENSION_TO_CONTENT_TYPE[ext]

        # Default to HTML for web pages
        return "text/html"

    def _fetch_url(self, url: str) -> CrawlResult:
        """
        Fetch a single URL.

        Args:
            url: URL to fetch

        Returns:
            CrawlResult with content or error
        """
        try:
            response = self._session.get(
                url,
                auth=self._auth,
                timeout=self.timeout,
                verify=self.verify_ssl,
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = self._get_content_type(response, url)
            content = response.content

            # Check if content type is supported
            if content_type not in SUPPORTED_CONTENT_TYPES:
                return CrawlResult(
                    url=url,
                    content_type=content_type,
                    content=b"",
                    error=f"Unsupported content type: {content_type}",
                    success=False,
                )

            result = CrawlResult(
                url=url,
                content_type=content_type,
                content=content,
                meta={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "final_url": response.url,
                },
            )

            # Extract links and title from HTML
            if result.is_html:
                html_text = content.decode("utf-8", errors="replace")
                result.links = LinkExtractor.extract_links(html_text, url)
                result.title = LinkExtractor.extract_title(html_text)

            return result

        except requests.exceptions.Timeout:
            return CrawlResult(
                url=url,
                content_type="",
                content=b"",
                error="Request timed out",
                success=False,
            )
        except requests.exceptions.SSLError as e:
            return CrawlResult(
                url=url,
                content_type="",
                content=b"",
                error=f"SSL error: {e}",
                success=False,
            )
        except requests.exceptions.RequestException as e:
            return CrawlResult(
                url=url,
                content_type="",
                content=b"",
                error=str(e),
                success=False,
            )

    def _save_to_temp_file(self, result: CrawlResult) -> str:
        """
        Save crawl result to a temporary file for processing.

        Args:
            result: CrawlResult with content

        Returns:
            Path to temporary file
        """
        # Determine file extension from content type
        ext_map = {
            "text/html": ".html",
            "application/xhtml+xml": ".html",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/pdf": ".pdf",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/json": ".json",
            "application/xml": ".xml",
            "text/xml": ".xml",
        }
        ext = ext_map.get(result.content_type, ".bin")

        # Create hash-based filename for deduplication
        url_hash = hashlib.md5(result.url.encode()).hexdigest()[:12]
        filename = f"crawl_{url_hash}{ext}"

        # Use configured temp dir or system temp
        if self.temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)
            filepath = os.path.join(self.temp_dir, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), "agentic_rag_crawl", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(result.content)

        return filepath

    def crawl(
        self,
        seed_urls: List[str],
        save_files: bool = True,
    ) -> Tuple[List[CrawlResult], CrawlStats]:
        """
        Crawl starting from seed URLs.

        Args:
            seed_urls: Initial URLs to crawl
            save_files: Whether to save content to temp files

        Returns:
            Tuple of (list of CrawlResults, CrawlStats)
        """
        stats = CrawlStats()
        results: List[CrawlResult] = []

        # Initialize queue with seed URLs at depth 0
        queue: List[Tuple[str, int]] = [(url, 0) for url in seed_urls]
        seed_domain = self._normalizer.get_domain(seed_urls[0]) if seed_urls else ""

        # Track discovered URLs
        discovered: Set[str] = set(self._normalizer.normalize(url) for url in seed_urls)
        stats.urls_discovered = len(discovered)

        logger.info(f"Starting crawl with {len(seed_urls)} seed URLs, max_depth={self.max_depth}")

        while queue and len(results) < self.max_pages:
            url, depth = queue.pop(0)
            normalized = self._normalizer.normalize(url)

            # Skip if already visited
            if normalized in self._visited:
                continue

            # Check depth limit
            if depth > self.max_depth:
                stats.urls_skipped += 1
                continue

            # Check filters
            if not self._should_crawl(url, seed_domain):
                stats.urls_skipped += 1
                continue

            # Mark as visited
            self._visited.add(normalized)

            # Fetch URL
            logger.debug(f"Crawling (depth={depth}): {url}")
            result = self._fetch_url(url)

            if result.success:
                stats.urls_crawled += 1
                stats.bytes_downloaded += len(result.content)
                stats.pages_by_depth[depth] = stats.pages_by_depth.get(depth, 0) + 1
                stats.content_types[result.content_type] = (
                    stats.content_types.get(result.content_type, 0) + 1
                )

                # Save to temp file if requested
                if save_files and result.content:
                    result.local_path = self._save_to_temp_file(result)
                    result.meta["source_url"] = url
                    result.meta["crawl_depth"] = depth

                results.append(result)

                # Queue discovered links (if not at max depth)
                if depth < self.max_depth and result.links:
                    for link in result.links:
                        link_normalized = self._normalizer.normalize(link)
                        if link_normalized not in discovered:
                            discovered.add(link_normalized)
                            stats.urls_discovered += 1
                            queue.append((link, depth + 1))

            else:
                stats.urls_failed += 1
                stats.errors.append((url, result.error or "Unknown error"))
                logger.warning(f"Failed to crawl {url}: {result.error}")

            # Rate limiting
            if self.delay > 0 and queue:
                time.sleep(self.delay)

        logger.info(
            f"Crawl complete: {stats.urls_crawled} pages crawled, "
            f"{stats.urls_failed} failed, {stats.bytes_downloaded} bytes"
        )

        return results, stats

    def crawl_single(self, url: str, save_file: bool = True) -> CrawlResult:
        """
        Crawl a single URL without following links.

        Args:
            url: URL to crawl
            save_file: Whether to save content to temp file

        Returns:
            CrawlResult
        """
        result = self._fetch_url(url)

        if result.success and save_file and result.content:
            result.local_path = self._save_to_temp_file(result)
            result.meta["source_url"] = url
            result.meta["crawl_depth"] = 0

        return result

    def close(self) -> None:
        """Close the session and clean up resources."""
        self._session.close()

    def __enter__(self) -> "WebCrawler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def crawl_urls(
    urls: List[str],
    max_depth: int = 2,
    max_pages: int = 100,
    same_domain_only: bool = True,
    basic_auth: Optional[Tuple[str, str]] = None,
    **kwargs,
) -> Tuple[List[CrawlResult], CrawlStats]:
    """
    Convenience function to crawl URLs.

    Args:
        urls: Seed URLs to crawl
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl
        same_domain_only: Only crawl same domain
        basic_auth: Optional (username, password)
        **kwargs: Additional arguments for WebCrawler

    Returns:
        Tuple of (results, stats)
    """
    with WebCrawler(
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        basic_auth=basic_auth,
        **kwargs,
    ) as crawler:
        return crawler.crawl(urls)
