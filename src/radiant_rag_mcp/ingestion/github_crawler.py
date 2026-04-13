"""
GitHub-aware crawler for repository content ingestion.

Handles GitHub repositories specially by:
1. Detecting GitHub repository URLs
2. Using raw.githubusercontent.com for actual content
3. Parsing README to find linked markdown files
4. Fetching and processing all markdown content
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


@dataclass
class GitHubFile:
    """Represents a file in a GitHub repository."""
    
    path: str
    name: str
    url: str
    raw_url: str
    content: Optional[str] = None
    size: int = 0
    sha: Optional[str] = None


@dataclass
class GitHubRepo:
    """Represents a GitHub repository."""
    
    owner: str
    repo: str
    branch: str = "main"
    
    @property
    def api_url(self) -> str:
        """GitHub API URL for this repo."""
        return f"https://api.github.com/repos/{self.owner}/{self.repo}"
    
    @property
    def raw_base_url(self) -> str:
        """Base URL for raw content."""
        return f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}"
    
    @property
    def html_url(self) -> str:
        """HTML URL for this repo."""
        return f"https://github.com/{self.owner}/{self.repo}"


@dataclass
class GitHubCrawlResult:
    """Result from crawling a GitHub repository."""
    
    repo: GitHubRepo
    files: List[GitHubFile] = field(default_factory=list)
    readme_content: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class GitHubCrawler:
    """
    Crawler specifically designed for GitHub repositories.
    
    Features:
    - Detects GitHub URLs and extracts owner/repo
    - Fetches raw content from raw.githubusercontent.com
    - Parses README to find linked markdown files
    - Recursively fetches all markdown content
    - Handles rate limiting gracefully
    """
    
    # Pattern to match GitHub repo URLs (captures before any query string)
    GITHUB_URL_PATTERN = re.compile(
        r"https?://(?:www\.)?github\.com/([^/]+)/([^/?#]+)/?(?:tree/([^/?#]+))?(?:/([^?#]*))?(?:[?#].*)?"
    )
    
    # Pattern to match markdown links in content
    MARKDOWN_LINK_PATTERN = re.compile(
        r"\[([^\]]+)\]\(([^)]+\.md)(?:#[^)]+)?\)"
    )
    
    # Pattern to match GitHub blob URLs
    BLOB_URL_PATTERN = re.compile(
        r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
    )
    
    def __init__(
        self,
        timeout: int = 30,
        delay: float = 0.5,
        max_files: int = 200,
        include_extensions: Optional[List[str]] = None,
        github_token: Optional[str] = None,
    ):
        """
        Initialize the GitHub crawler.
        
        Args:
            timeout: Request timeout in seconds
            delay: Delay between requests (rate limiting)
            max_files: Maximum files to fetch
            include_extensions: File extensions to include (default: docs + code)
            github_token: Optional GitHub API token for higher rate limits
        """
        self.timeout = timeout
        self.delay = delay
        self.max_files = max_files
        
        # Default extensions: documentation + code files
        self.include_extensions = include_extensions or [
            # Documentation
            ".md", ".txt", ".rst", ".mdx",
            # Python
            ".py", ".pyw", ".pyx",
            # JavaScript/TypeScript
            ".js", ".jsx", ".ts", ".tsx", ".mjs",
            # Java/JVM
            ".java", ".kt", ".kts", ".scala",
            # Systems
            ".go", ".rs", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs",
            # Scripting
            ".rb", ".php", ".swift", ".r",
            # Shell
            ".sh", ".bash", ".zsh",
            # Config/Data
            ".yaml", ".yml", ".json", ".toml",
            # SQL
            ".sql",
        ]
        
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "RadiantRAG-GitHubCrawler/1.0",
            "Accept": "application/vnd.github.v3+json",
        })
        
        if self.github_token:
            self._session.headers["Authorization"] = f"token {self.github_token}"
    
    @classmethod
    def is_github_url(cls, url: str) -> bool:
        """Check if a URL is a GitHub repository URL."""
        return bool(cls.GITHUB_URL_PATTERN.match(url))
    
    @classmethod
    def parse_github_url(cls, url: str) -> Optional[GitHubRepo]:
        """
        Parse a GitHub URL into owner, repo, and branch.
        
        Args:
            url: GitHub URL
            
        Returns:
            GitHubRepo object or None if not a valid GitHub URL
        """
        # Parse the URL (local import to avoid unused top-level import)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Note: parsed is used for future extension (clean URL handling)
        _ = parsed  # Acknowledge parsed for potential future use
        
        match = cls.GITHUB_URL_PATTERN.match(url)
        if not match:
            return None
        
        owner, repo, branch, path = match.groups()
        
        # Clean repo name (remove .git suffix if present)
        repo = repo.removesuffix(".git")
        
        # Remove any remaining query string artifacts
        if "?" in repo:
            repo = repo.split("?")[0]
        
        # Default branch
        branch = branch or "main"
        
        return GitHubRepo(owner=owner, repo=repo, branch=branch)
    
    def _fetch_raw_content(self, url: str) -> Optional[str]:
        """
        Fetch raw content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Content as string or None on error
        """
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def _convert_to_raw_url(self, repo: GitHubRepo, path: str) -> str:
        """Convert a file path to raw.githubusercontent.com URL."""
        # Clean path
        path = path.lstrip("/")
        return f"{repo.raw_base_url}/{path}"
    
    def _convert_blob_to_raw_url(self, blob_url: str) -> Optional[str]:
        """
        Convert a GitHub blob URL to a raw content URL.
        
        Example:
            Input:  https://github.com/owner/repo/blob/main/path/file.md
            Output: https://raw.githubusercontent.com/owner/repo/main/path/file.md
        """
        match = self.BLOB_URL_PATTERN.match(blob_url)
        if not match:
            return None
        
        owner, repo, branch, path = match.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    
    def _extract_markdown_links(
        self,
        content: str,
        base_path: str = "",
    ) -> List[str]:
        """
        Extract markdown file links from content.
        
        Args:
            content: Markdown content
            base_path: Base path for relative links
            
        Returns:
            List of relative file paths
        """
        links = set()
        
        # Find all markdown links
        for match in self.MARKDOWN_LINK_PATTERN.finditer(content):
            link_text, link_url = match.groups()
            
            # Skip external links
            if link_url.startswith(("http://", "https://")):
                # Check if it's a GitHub blob link to the same repo
                raw_url = self._convert_blob_to_raw_url(link_url)
                if raw_url:
                    # Extract just the path portion
                    match2 = self.BLOB_URL_PATTERN.match(link_url)
                    if match2:
                        path = match2.group(4)
                        links.add(path)
                continue
            
            # Skip anchors only
            if link_url.startswith("#"):
                continue
            
            # Handle relative paths
            if base_path:
                # Resolve relative to base path
                base_dir = os.path.dirname(base_path)
                resolved = os.path.normpath(os.path.join(base_dir, link_url))
            else:
                resolved = link_url
            
            # Clean path
            resolved = resolved.lstrip("./")
            
            links.add(resolved)
        
        return list(links)
    
    def _list_repo_files_api(self, repo: GitHubRepo, path: str = "") -> List[GitHubFile]:
        """
        List files in a repository using GitHub API.
        
        Args:
            repo: GitHub repository
            path: Path within repository
            
        Returns:
            List of GitHubFile objects
        """
        url = f"{repo.api_url}/contents/{path}"
        
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            items = response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to list files at {url}: {e}")
            return []
        
        files = []
        
        # Handle error responses from API
        if isinstance(items, dict):
            if "message" in items:
                # API error response
                logger.warning(f"GitHub API error: {items.get('message')}")
                return []
            # Single file
            items = [items]
        
        for item in items:
            # Skip items without type (malformed response)
            if not isinstance(item, dict) or "type" not in item:
                logger.warning(f"Skipping malformed item in API response: {item}")
                continue
                
            if item["type"] == "file":
                ext = os.path.splitext(item["name"])[1].lower()
                if ext in self.include_extensions:
                    files.append(GitHubFile(
                        path=item["path"],
                        name=item["name"],
                        url=item["html_url"],
                        raw_url=item.get("download_url", ""),
                        size=item.get("size", 0),
                        sha=item.get("sha"),
                    ))
            elif item["type"] == "dir":
                # Recursively list directory contents
                time.sleep(self.delay)  # Rate limiting
                sub_files = self._list_repo_files_api(repo, item["path"])
                files.extend(sub_files)
                
                if len(files) >= self.max_files:
                    break
        
        return files[:self.max_files]
    
    def _fetch_readme(self, repo: GitHubRepo) -> Optional[str]:
        """
        Fetch the README file content.
        
        Args:
            repo: GitHub repository
            
        Returns:
            README content or None
        """
        # Try common README filenames
        readme_names = ["README.md", "readme.md", "README.MD", "Readme.md"]
        
        for name in readme_names:
            url = self._convert_to_raw_url(repo, name)
            content = self._fetch_raw_content(url)
            if content:
                return content
        
        return None
    
    def crawl(
        self,
        url: str,
        fetch_all_files: bool = False,
        follow_readme_links: bool = True,
    ) -> GitHubCrawlResult:
        """
        Crawl a GitHub repository.
        
        Args:
            url: GitHub repository URL
            fetch_all_files: If True, fetch all markdown files in repo
            follow_readme_links: If True, follow links found in README
            
        Returns:
            GitHubCrawlResult with fetched files
        """
        repo = self.parse_github_url(url)
        if not repo:
            return GitHubCrawlResult(
                repo=GitHubRepo(owner="", repo=""),
                errors=[f"Invalid GitHub URL: {url}"],
            )
        
        result = GitHubCrawlResult(repo=repo)
        fetched_paths: Set[str] = set()
        
        logger.info(f"Crawling GitHub repo: {repo.owner}/{repo.repo}")
        
        # 1. Fetch README
        readme = self._fetch_readme(repo)
        if readme:
            result.readme_content = readme
            result.files.append(GitHubFile(
                path="README.md",
                name="README.md",
                url=f"{repo.html_url}/blob/{repo.branch}/README.md",
                raw_url=self._convert_to_raw_url(repo, "README.md"),
                content=readme,
            ))
            fetched_paths.add("README.md")
            result.stats["readme_fetched"] = True
            
            # Extract links from README
            if follow_readme_links:
                linked_paths = self._extract_markdown_links(readme, "")
                logger.info(f"Found {len(linked_paths)} linked files in README")
                result.stats["readme_links_found"] = len(linked_paths)
        
        time.sleep(self.delay)
        
        # 2. Determine which files to fetch
        files_to_fetch: List[str] = []
        
        if fetch_all_files:
            # Use API to list all files
            logger.info("Listing all repository files...")
            all_files = self._list_repo_files_api(repo)
            files_to_fetch = [f.path for f in all_files if f.path not in fetched_paths]
            result.stats["api_files_found"] = len(all_files)
        elif follow_readme_links and readme:
            # Only fetch files linked from README
            linked_paths = self._extract_markdown_links(readme, "")
            files_to_fetch = [p for p in linked_paths if p not in fetched_paths]
        
        # 3. Fetch each file
        logger.info(f"Fetching {len(files_to_fetch)} files...")
        
        for path in files_to_fetch[:self.max_files]:
            if path in fetched_paths:
                continue
            
            raw_url = self._convert_to_raw_url(repo, path)
            content = self._fetch_raw_content(raw_url)
            
            if content:
                result.files.append(GitHubFile(
                    path=path,
                    name=os.path.basename(path),
                    url=f"{repo.html_url}/blob/{repo.branch}/{path}",
                    raw_url=raw_url,
                    content=content,
                ))
                fetched_paths.add(path)
                logger.debug(f"Fetched: {path}")
                
                # Follow links in this file too
                if follow_readme_links:
                    sub_links = self._extract_markdown_links(content, path)
                    for sub_path in sub_links:
                        if sub_path not in fetched_paths and sub_path not in files_to_fetch:
                            files_to_fetch.append(sub_path)
            else:
                result.errors.append(f"Failed to fetch: {path}")
            
            time.sleep(self.delay)
            
            if len(result.files) >= self.max_files:
                break
        
        result.stats["files_fetched"] = len(result.files)
        result.stats["errors"] = len(result.errors)
        
        logger.info(
            f"Crawl complete: {len(result.files)} files fetched, "
            f"{len(result.errors)} errors"
        )
        
        return result
    
    def save_to_files(
        self,
        result: GitHubCrawlResult,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Save crawled content to temporary files.
        
        Args:
            result: GitHubCrawlResult
            output_dir: Output directory (uses temp if not specified)
            
        Returns:
            List of saved file paths
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = tempfile.mkdtemp(prefix="github_crawl_")
        
        saved_paths = []
        
        for file in result.files:
            if not file.content:
                continue
            
            # Create safe filename
            safe_name = file.path.replace("/", "_").replace("\\", "_")
            filepath = os.path.join(output_dir, safe_name)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file.content)
            
            saved_paths.append(filepath)
        
        return saved_paths
    
    def close(self) -> None:
        """Close the session."""
        self._session.close()
    
    def __enter__(self) -> "GitHubCrawler":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def crawl_github_repo(
    url: str,
    fetch_all_files: bool = False,
    follow_readme_links: bool = True,
    github_token: Optional[str] = None,
) -> GitHubCrawlResult:
    """
    Convenience function to crawl a GitHub repository.
    
    Args:
        url: GitHub repository URL
        fetch_all_files: Fetch all markdown files (uses API)
        follow_readme_links: Follow links found in README
        github_token: Optional GitHub API token
        
    Returns:
        GitHubCrawlResult
    """
    with GitHubCrawler(github_token=github_token) as crawler:
        return crawler.crawl(
            url,
            fetch_all_files=fetch_all_files,
            follow_readme_links=follow_readme_links,
        )
