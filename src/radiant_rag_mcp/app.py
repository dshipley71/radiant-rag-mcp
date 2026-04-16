"""
Main application for Radiant Agentic RAG.

Provides the complete RAG system with document ingestion and querying.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from radiant_rag_mcp.config import AppConfig, load_config, setup_logging
from radiant_rag_mcp.utils.metrics import MetricsCollector
from radiant_rag_mcp.storage import create_vector_store, StoredDoc
from radiant_rag_mcp.storage.bm25_index import PersistentBM25Index
from radiant_rag_mcp.utils.conversation import ConversationManager, ConversationStore
from radiant_rag_mcp.llm.client import LLMClients
from radiant_rag_mcp.ingestion.processor import (
    DocumentProcessor,
    IngestedChunk,
    TranslatingDocumentProcessor,
)
from radiant_rag_mcp.ingestion.github_crawler import GitHubCrawler
from radiant_rag_mcp.orchestrator import PipelineResult, RAGOrchestrator, SimplifiedOrchestrator
from radiant_rag_mcp.ingestion.image_captioner import VLMConfig, create_captioner
from radiant_rag_mcp.agents.language_detection import LanguageDetectionAgent
from radiant_rag_mcp.agents.translation import TranslationAgent
from radiant_rag_mcp.ui.display import (
    console,
    display_error,
    display_index_stats,
    display_info,
    display_success,
    ProgressDisplay,
)
from radiant_rag_mcp.ui.reports.report import (
    QueryReport,
    display_search_results,
    print_report,
    save_report,
    save_search_report,
)

logger = logging.getLogger(__name__)


class RadiantRAG:
    """
    Main Radiant RAG application.
    
    Provides document ingestion and querying with the full agentic pipeline.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Initialize Radiant RAG.
        
        Args:
            config: Optional pre-loaded configuration
            config_path: Optional path to config file
        """
        # Load configuration
        self._config = config or load_config(config_path)
        
        # Setup logging
        setup_logging(self._config.logging)
        
        logger.info("Initializing Radiant RAG...")

        # Initialize components
        self._llm_clients = LLMClients.build(
            self._config.ollama,
            self._config.local_models,
            self._config.parsing,
            embedding_cache_size=self._config.performance.embedding_cache_size,
            # New backend configurations (optional, uses old config if not provided)
            llm_backend_config=self._config.llm_backend,
            embedding_backend_config=self._config.embedding_backend,
            reranking_backend_config=self._config.reranking_backend,
        )
        
        # Create vector store using factory (supports redis, chroma, pgvector)
        self._store = create_vector_store(self._config)
        
        self._bm25_index = PersistentBM25Index(
            self._config.bm25,
            self._store,
        )
        
        # Conversation management
        if self._config.conversation.enabled:
            conv_store = ConversationStore(
                self._config.redis,
                self._config.conversation,
            )
            self._conversation_manager = ConversationManager(
                conv_store,
                self._config.conversation,
            )
        else:
            self._conversation_manager = None

        # Initialize orchestrator
        self._orchestrator = RAGOrchestrator(
            config=self._config,
            llm=self._llm_clients.chat,
            local=self._llm_clients.local,
            store=self._store,
            bm25_index=self._bm25_index,
            conversation_manager=self._conversation_manager,
        )
        
        # Create image captioner for VLM-based image processing (HuggingFace Transformers)
        vlm_config = VLMConfig(
            model_name=self._config.vlm.model_name,
            device=self._config.vlm.device,
            load_in_4bit=self._config.vlm.load_in_4bit,
            load_in_8bit=self._config.vlm.load_in_8bit,
            max_new_tokens=self._config.vlm.max_new_tokens,
            temperature=self._config.vlm.temperature,
            enabled=self._config.vlm.enabled,
            cache_dir=self._config.vlm.cache_dir,
        )
        self._image_captioner = create_captioner(
            vlm_config=vlm_config,
            ollama_url=self._config.vlm.ollama_fallback_url,
            ollama_model=self._config.vlm.ollama_fallback_model,
        )
        
        # Document processor
        self._base_doc_processor = DocumentProcessor(
            self._config.unstructured_cleaning,
            image_captioner=self._image_captioner,
        )
        
        # Initialize language detection and translation agents if enabled
        self._lang_detection_agent = None
        self._translation_agent = None
        
        if self._config.language_detection.enabled:
            self._lang_detection_agent = LanguageDetectionAgent(
                llm=self._llm_clients.chat if self._config.language_detection.use_llm_fallback else None,
                method=self._config.language_detection.method,
                min_confidence=self._config.language_detection.min_confidence,
                use_llm_fallback=self._config.language_detection.use_llm_fallback,
                fallback_language=self._config.language_detection.fallback_language,
                model_path=self._config.language_detection.model_path,
                model_url=self._config.language_detection.model_url,
                auto_download=self._config.language_detection.auto_download,
                verify_checksum=self._config.language_detection.verify_checksum,
            )
            logger.info(
                f"Language detection enabled (method={self._config.language_detection.method})"
            )
        
        if self._config.translation.enabled:
            self._translation_agent = TranslationAgent(
                llm=self._llm_clients.chat,
                canonical_language=self._config.translation.canonical_language,
                max_chars_per_call=self._config.translation.max_chars_per_llm_call,
                preserve_original=self._config.translation.preserve_original,
            )
            logger.info(
                f"Translation enabled (canonical_language={self._config.translation.canonical_language})"
            )
        
        # Create translating processor if both agents are available
        if (self._config.translation.translate_at_ingestion and 
            self._lang_detection_agent is not None and 
            self._translation_agent is not None):
            self._doc_processor = TranslatingDocumentProcessor(
                base_processor=self._base_doc_processor,
                language_detection_agent=self._lang_detection_agent,
                translation_agent=self._translation_agent,
                canonical_language=self._config.translation.canonical_language,
                translate_at_ingestion=True,
                preserve_original=self._config.translation.preserve_original,
            )
            logger.info("Using TranslatingDocumentProcessor for ingestion")
        else:
            self._doc_processor = self._base_doc_processor

        # Metrics collector
        self._metrics_collector = MetricsCollector(
            max_history=self._config.metrics.history_retention
        )

        logger.info("Radiant RAG initialized successfully")

    @property
    def config(self) -> AppConfig:
        """Get configuration."""
        return self._config

    @property
    def store(self):
        """Get vector store (Redis, Chroma, or PgVector)."""
        return self._store

    @property
    def bm25_index(self) -> PersistentBM25Index:
        """Get BM25 index."""
        return self._bm25_index

    def check_health(self) -> Dict[str, Any]:
        """
        Check system health.
        
        Returns:
            Health status dictionary
        """
        health = {
            "redis": False,
            "vector_index": {},
            "bm25_index": {},
            "conversation": self._conversation_manager is not None,
        }

        # Check Redis connection
        try:
            health["redis"] = self._store.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        # Get index stats
        try:
            health["vector_index"] = self._store.get_index_info()
        except Exception as e:
            logger.error(f"Vector index check failed: {e}")

        try:
            health["bm25_index"] = self._bm25_index.get_stats()
        except Exception as e:
            logger.error(f"BM25 index check failed: {e}")

        return health

    def ingest_documents(
        self,
        paths: Sequence[str],
        show_progress: bool = True,
        use_hierarchical: bool = True,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            paths: File or directory paths to ingest
            show_progress: Whether to show progress
            use_hierarchical: Whether to use hierarchical (parent/child) storage
            child_chunk_size: Size of child chunks
            child_chunk_overlap: Overlap between child chunks
            
        Returns:
            Ingestion statistics
        """
        stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "documents_stored": 0,
            "errors": [],
        }

        if show_progress:
            progress = ProgressDisplay("Processing documents...")
            progress.__enter__()

        try:
            # Process all paths
            results = self._doc_processor.process_paths(
                paths,
                split_large_chunks=use_hierarchical,
                max_chunk_size=child_chunk_size * 2,
            )

            for file_path, chunks in results.items():
                if show_progress:
                    progress.update(f"Processing: {Path(file_path).name}")

                if not chunks:
                    stats["files_failed"] += 1
                    stats["errors"].append(f"No chunks from: {file_path}")
                    continue

                stats["files_processed"] += 1
                stats["chunks_created"] += len(chunks)

                try:
                    if use_hierarchical:
                        stored = self._ingest_hierarchical(
                            file_path, chunks,
                            child_chunk_size, child_chunk_overlap
                        )
                    else:
                        stored = self._ingest_flat(chunks)
                    
                    stats["documents_stored"] += stored
                except Exception as e:
                    logger.error(f"Failed to store documents from {file_path}: {e}")
                    stats["errors"].append(f"Storage failed for {file_path}: {e}")

            # Sync BM25 index
            if show_progress:
                progress.update("Syncing BM25 index...")
            
            self._bm25_index.sync_with_store()
            self._bm25_index.save()

        finally:
            if show_progress:
                progress.__exit__(None, None, None)

        return stats

    def ingest_videos(
        self,
        sources: List[str],
        show_progress: bool = True,
        use_hierarchical: bool = True,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 50,
        enable_frame_captioning: bool = False,
        force_frame_analysis: bool = False,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        """Ingest videos into the RAG system."""
        from radiant_rag_mcp.ingestion.video_processor import VideoProcessor
        from dataclasses import replace

        video_cfg = self._config.video
        if enable_frame_captioning != video_cfg.enable_frame_captioning:
            video_cfg = replace(video_cfg, enable_frame_captioning=enable_frame_captioning)

        captioner = self._image_captioner if (
            enable_frame_captioning or video_cfg.enable_silent_video_analysis
        ) else None

        processor = VideoProcessor(config=video_cfg, image_captioner=captioner)
        stats: Dict[str, Any] = {
            "sources_processed": 0,
            "sources_failed": 0,
            "chunks_created": 0,
            "documents_stored": 0,
            "silent_sources": 0,
            "audio_sources": 0,
            "summaries": {},
            "errors": [],
        }

        for source in sources:
            try:
                chunks = processor.process_video(source,
                            force_frame_analysis=force_frame_analysis)
                if not chunks:
                    stats["sources_failed"] += 1
                    stats["errors"].append(f"No chunks from: {source}")
                    continue
                ctypes = {c.meta.get("content_type") for c in chunks}
                if "frame_window_captions" in ctypes:
                    stats["silent_sources"] += 1
                if "transcript" in ctypes:
                    stats["audio_sources"] += 1
                stats["sources_processed"] += 1
                stats["chunks_created"] += len(chunks)
                stored = (self._ingest_hierarchical(source, chunks,
                              child_chunk_size, child_chunk_overlap)
                          if use_hierarchical else self._ingest_flat(chunks))
                stats["documents_stored"] += stored
                if summarize:
                    from radiant_rag_mcp.agents.video_summarization import VideoSummarizationAgent
                    agent = VideoSummarizationAgent(
                        llm=self._llm_clients.chat,
                        config=self._config.video_summarization)
                    result = agent.summarize_video(source, chunks)
                    stats["summaries"][source] = result.__dict__
            except Exception as e:
                logger.error(f"Failed to ingest video {source}: {e}")
                stats["sources_failed"] += 1
                stats["errors"].append(f"{source}: {e}")

        self._bm25_index.sync_with_store()
        self._bm25_index.save()
        return stats

    def _ingest_flat(self, chunks: List[IngestedChunk]) -> int:
        """Ingest chunks without hierarchical structure using batch processing."""
        if not chunks:
            return 0

        ingestion_cfg = self._config.ingestion
        embed_batch_size = ingestion_cfg.embedding_batch_size
        redis_batch_size = ingestion_cfg.redis_batch_size
        stored = 0

        # Prepare all texts for batch embedding
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batches
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), embed_batch_size):
            batch_texts = texts[i:i + embed_batch_size]
            batch_embeddings = self._llm_clients.local.embed(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Prepare documents for batch upsert
        documents = []
        for chunk, embedding in zip(chunks, all_embeddings):
            doc_id = self._store.make_doc_id(chunk.content, chunk.meta)
            documents.append({
                "doc_id": doc_id,
                "content": chunk.content,
                "embedding": embedding,
                "meta": {**chunk.meta, "doc_level": "child"},
            })

        # Upsert in batches
        for i in range(0, len(documents), redis_batch_size):
            batch = documents[i:i + redis_batch_size]
            stored += self._store.upsert_batch(batch)

        return stored

    def _ingest_hierarchical(
        self,
        file_path: str,
        chunks: List[IngestedChunk],
        child_size: int,
        child_overlap: int,
    ) -> int:
        """Ingest chunks with hierarchical (parent/child) structure.
        
        When embed_parents is enabled in config, parent documents are also
        embedded and can be retrieved via similarity search. Otherwise,
        only child chunks are embedded (default behavior).
        """
        from radiant_rag_mcp.ingestion.processor import ChunkSplitter
        
        if not chunks:
            return 0

        ingestion_cfg = self._config.ingestion
        splitter = ChunkSplitter(child_size, child_overlap)
        embed_parents = ingestion_cfg.embed_parents

        # Batch mode (always enabled for performance)
        embed_batch_size = ingestion_cfg.embedding_batch_size
        redis_batch_size = ingestion_cfg.redis_batch_size
        stored = 0

        # First pass: collect all parent docs and child info
        parent_docs = []
        child_info = []  # List of (child_id, child_text, child_meta)

        for chunk in chunks:
            parent_meta = {**chunk.meta, "doc_level": "parent"}
            parent_id = self._store.make_doc_id(chunk.content, parent_meta)
            
            parent_docs.append({
                "doc_id": parent_id,
                "content": chunk.content,
                "meta": parent_meta,
            })

            child_texts = splitter.split(chunk.content)
            for i, child_text in enumerate(child_texts):
                child_meta = {
                    **chunk.meta,
                    "doc_level": "child",
                    "parent_id": parent_id,
                    "child_index": i,
                    "child_total": len(child_texts),
                }
                child_id = self._store.make_doc_id(child_text, child_meta)
                child_info.append((child_id, child_text, child_meta))

        # Handle parent documents
        if embed_parents:
            # Generate embeddings for parents
            parent_texts = [p["content"] for p in parent_docs]
            parent_embeddings: List[List[float]] = []
            
            for i in range(0, len(parent_texts), embed_batch_size):
                batch_texts = parent_texts[i:i + embed_batch_size]
                batch_embeddings = self._llm_clients.local.embed(batch_texts)
                parent_embeddings.extend(batch_embeddings)
            
            # Add embeddings to parent docs
            for doc, embedding in zip(parent_docs, parent_embeddings):
                doc["embedding"] = embedding
            
            # Batch upsert parent documents with embeddings
            for i in range(0, len(parent_docs), redis_batch_size):
                batch = parent_docs[i:i + redis_batch_size]
                stored += self._store.upsert_batch(batch)
        else:
            # Batch upsert parent documents without embeddings (original behavior)
            for i in range(0, len(parent_docs), redis_batch_size):
                batch = parent_docs[i:i + redis_batch_size]
                stored += self._store.upsert_doc_only_batch(batch)

        # Generate embeddings for all children in batches
        child_texts_only = [c[1] for c in child_info]
        all_embeddings: List[List[float]] = []
        
        for i in range(0, len(child_texts_only), embed_batch_size):
            batch_texts = child_texts_only[i:i + embed_batch_size]
            batch_embeddings = self._llm_clients.local.embed(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Prepare child documents for batch upsert
        child_docs = []
        for (child_id, child_text, child_meta), embedding in zip(child_info, all_embeddings):
            child_docs.append({
                "doc_id": child_id,
                "content": child_text,
                "embedding": embedding,
                "meta": child_meta,
            })

        # Batch upsert child documents
        for i in range(0, len(child_docs), redis_batch_size):
            batch = child_docs[i:i + redis_batch_size]
            stored += self._store.upsert_batch(batch)

        return stored

    def ingest_urls(
        self,
        urls: Sequence[str],
        show_progress: bool = True,
        use_hierarchical: bool = True,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 50,
        crawl_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        basic_auth: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest documents from URLs into the RAG system.
        
        Automatically detects GitHub repository URLs and uses specialized
        GitHub crawling for proper content extraction.
        
        Args:
            urls: URLs to ingest (will be crawled based on config)
            show_progress: Whether to show progress
            use_hierarchical: Whether to use hierarchical (parent/child) storage
            child_chunk_size: Size of child chunks
            child_chunk_overlap: Overlap between child chunks
            crawl_depth: Override config crawl depth (0 = no crawling)
            max_pages: Override config max pages
            basic_auth: Override config basic auth (username, password)
            
        Returns:
            Ingestion statistics
        """
        from radiant_rag_mcp.ingestion.web_crawler import WebCrawler
        
        # Separate GitHub URLs from regular URLs
        github_urls = [u for u in urls if GitHubCrawler.is_github_url(u)]
        regular_urls = [u for u in urls if not GitHubCrawler.is_github_url(u)]
        
        stats = {
            "urls_crawled": 0,
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "documents_stored": 0,
            "github_repos": 0,
            "errors": [],
        }
        
        # Handle GitHub URLs with specialized crawler
        for github_url in github_urls:
            if show_progress:
                display_info(f"Detected GitHub repository: {github_url}")
            
            github_stats = self.ingest_github(
                github_url,
                show_progress=show_progress,
                use_hierarchical=use_hierarchical,
                child_chunk_size=child_chunk_size,
                child_chunk_overlap=child_chunk_overlap,
                fetch_all_files=True,  # Fetch all markdown files
                follow_readme_links=True,
            )
            
            stats["github_repos"] += 1
            stats["files_processed"] += github_stats.get("files_processed", 0)
            stats["chunks_created"] += github_stats.get("chunks_created", 0)
            stats["documents_stored"] += github_stats.get("documents_stored", 0)
            stats["errors"].extend(github_stats.get("errors", []))
        
        # If no regular URLs, return now
        if not regular_urls:
            return stats

        if show_progress:
            progress = ProgressDisplay("Crawling URLs...")
            progress.__enter__()

        try:
            # Configure crawler
            crawler_config = self._config.web_crawler
            
            # Apply overrides
            effective_depth = crawl_depth if crawl_depth is not None else crawler_config.max_depth
            effective_max_pages = max_pages if max_pages is not None else crawler_config.max_pages
            
            # Set up auth
            effective_auth = basic_auth
            if not effective_auth and crawler_config.basic_auth_user:
                effective_auth = (
                    crawler_config.basic_auth_user,
                    crawler_config.basic_auth_password,
                )

            # Create crawler
            crawler = WebCrawler(
                max_depth=effective_depth,
                max_pages=effective_max_pages,
                same_domain_only=crawler_config.same_domain_only,
                include_patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                timeout=crawler_config.timeout,
                delay=crawler_config.delay,
                user_agent=crawler_config.user_agent,
                basic_auth=effective_auth,
                verify_ssl=crawler_config.verify_ssl,
                temp_dir=crawler_config.temp_dir,
            )

            # Crawl URLs
            if show_progress:
                progress.update(f"Crawling {len(urls)} seed URLs (depth={effective_depth})...")
            
            results, crawl_stats = crawler.crawl(urls, save_files=True)
            stats["urls_crawled"] = crawl_stats.urls_crawled
            
            logger.info(
                f"Crawled {crawl_stats.urls_crawled} pages, "
                f"{crawl_stats.urls_failed} failed, "
                f"{crawl_stats.bytes_downloaded} bytes"
            )

            # Process crawled files
            for result in results:
                if not result.success or not result.local_path:
                    stats["files_failed"] += 1
                    if result.error:
                        stats["errors"].append(f"URL {result.url}: {result.error}")
                    continue

                if show_progress:
                    progress.update(f"Processing: {result.url[:50]}...")

                try:
                    # Process the downloaded file
                    chunks = self._doc_processor.process_file(
                        result.local_path,
                        split_large_chunks=use_hierarchical,
                        max_chunk_size=child_chunk_size * 2,
                    )

                    if not chunks:
                        stats["files_failed"] += 1
                        stats["errors"].append(f"No chunks from: {result.url}")
                        continue

                    # Add URL metadata to chunks
                    enhanced_chunks = []
                    for chunk in chunks:
                        enhanced_meta = {
                            **chunk.meta,
                            "source_url": result.url,
                            "source_type": "web",
                            "content_type": result.content_type,
                            "page_title": result.title,
                            "crawl_depth": result.meta.get("crawl_depth", 0),
                        }
                        enhanced_chunks.append(
                            IngestedChunk(content=chunk.content, meta=enhanced_meta)
                        )

                    stats["files_processed"] += 1
                    stats["chunks_created"] += len(enhanced_chunks)

                    # Store chunks
                    if use_hierarchical:
                        stored = self._ingest_hierarchical(
                            result.url,
                            enhanced_chunks,
                            child_chunk_size,
                            child_chunk_overlap,
                        )
                    else:
                        stored = self._ingest_flat(enhanced_chunks)

                    stats["documents_stored"] += stored

                except Exception as e:
                    logger.error(f"Failed to process {result.url}: {e}")
                    stats["errors"].append(f"Processing failed for {result.url}: {e}")
                    stats["files_failed"] += 1

            # Clean up temp files
            for result in results:
                if result.local_path and os.path.exists(result.local_path):
                    try:
                        os.remove(result.local_path)
                    except Exception:
                        pass

            # Sync BM25 index
            if show_progress:
                progress.update("Syncing BM25 index...")

            self._bm25_index.sync_with_store()
            self._bm25_index.save()

            crawler.close()

        finally:
            if show_progress:
                progress.__exit__(None, None, None)

        return stats

    def ingest_github(
        self,
        url: str,
        show_progress: bool = True,
        use_hierarchical: bool = True,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 50,
        fetch_all_files: bool = True,
        follow_readme_links: bool = True,
        github_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a GitHub repository into the RAG system.
        
        This method properly handles GitHub repositories by:
        1. Detecting it's a GitHub URL
        2. Fetching raw markdown content (not HTML)
        3. Following links in README to find all content files
        4. Indexing each markdown file as a document
        
        Args:
            url: GitHub repository URL
            show_progress: Whether to show progress
            use_hierarchical: Whether to use hierarchical storage
            child_chunk_size: Size of child chunks
            child_chunk_overlap: Overlap between chunks
            fetch_all_files: If True, fetch ALL markdown files using API
            follow_readme_links: If True, follow links found in README
            github_token: Optional GitHub API token for higher rate limits
            
        Returns:
            Ingestion statistics
        """
        stats = {
            "repo": "",
            "files_fetched": 0,
            "files_processed": 0,
            "chunks_created": 0,
            "documents_stored": 0,
            "errors": [],
        }
        
        # Check if it's a GitHub URL
        if not GitHubCrawler.is_github_url(url):
            stats["errors"].append(f"Not a GitHub URL: {url}")
            return stats
        
        if show_progress:
            progress = ProgressDisplay("Crawling GitHub repository...")
            progress.__enter__()
        
        try:
            # Create crawler
            crawler = GitHubCrawler(
                github_token=github_token,
                max_files=200,
            )
            
            if show_progress:
                progress.update(f"Fetching content from {url}...")
            
            # Crawl the repository
            result = crawler.crawl(
                url,
                fetch_all_files=fetch_all_files,
                follow_readme_links=follow_readme_links,
            )
            
            stats["repo"] = f"{result.repo.owner}/{result.repo.repo}"
            stats["files_fetched"] = len(result.files)
            
            logger.info(
                f"Fetched {len(result.files)} files from "
                f"{result.repo.owner}/{result.repo.repo}"
            )
            
            # Process each file
            for github_file in result.files:
                if not github_file.content:
                    continue
                
                if show_progress:
                    progress.update(f"Processing: {github_file.path}...")
                
                try:
                    # Create chunks from the file content (handles both code and markdown)
                    chunks = self._chunk_github_content(
                        github_file.content,
                        source_path=github_file.path,
                        source_url=github_file.url,
                        repo_name=f"{result.repo.owner}/{result.repo.repo}",
                    )
                    
                    if not chunks:
                        continue
                    
                    stats["files_processed"] += 1
                    stats["chunks_created"] += len(chunks)
                    
                    # Store chunks
                    if use_hierarchical:
                        stored = self._ingest_hierarchical(
                            github_file.path,
                            chunks,
                            child_chunk_size,
                            child_chunk_overlap,
                        )
                    else:
                        stored = self._ingest_flat(chunks)
                    
                    stats["documents_stored"] += stored
                    
                except Exception as e:
                    logger.error(f"Failed to process {github_file.path}: {e}")
                    stats["errors"].append(f"{github_file.path}: {e}")
            
            # Add any crawler errors
            stats["errors"].extend(result.errors)
            
            # Sync BM25 index
            if show_progress:
                progress.update("Syncing BM25 index...")
            
            self._bm25_index.sync_with_store()
            self._bm25_index.save()
            
            crawler.close()
            
        finally:
            if show_progress:
                progress.__exit__(None, None, None)
        
        return stats
    
    def _chunk_github_content(
        self,
        content: str,
        source_path: str,
        source_url: str,
        repo_name: str,
    ) -> List[IngestedChunk]:
        """
        Chunk GitHub file content intelligently.
        
        Handles both documentation and code files:
        - Markdown: Q&A patterns, headers, paragraphs
        - Code: Functions, classes, methods with imports context
        
        Args:
            content: File content
            source_path: Source file path
            source_url: Source URL
            repo_name: Repository name
            
        Returns:
            List of IngestedChunk objects
        """
        import os
        
        # Detect file type
        ext = os.path.splitext(source_path)[1].lower()
        
        # Code file extensions
        code_extensions = {
            ".py", ".pyw", ".pyx",  # Python
            ".js", ".jsx", ".ts", ".tsx", ".mjs",  # JavaScript/TypeScript
            ".java", ".kt", ".kts", ".scala",  # JVM
            ".go", ".rs", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs",  # Systems
            ".rb", ".php", ".swift", ".r",  # Scripting
            ".sh", ".bash", ".zsh",  # Shell
            ".sql",  # SQL
        }
        
        # Handle code files with code-aware chunking
        if ext in code_extensions:
            return self._chunk_code_content(content, source_path, source_url, repo_name)
        
        # Handle markdown/documentation files
        return self._chunk_markdown_content(content, source_path, source_url, repo_name)
    
    def _chunk_code_content(
        self,
        content: str,
        source_path: str,
        source_url: str,
        repo_name: str,
    ) -> List[IngestedChunk]:
        """
        Chunk code files using code-aware parsing.
        
        Args:
            content: Code content
            source_path: Source file path
            source_url: Source URL
            repo_name: Repository name
            
        Returns:
            List of IngestedChunk objects
        """
        from radiant_rag_mcp.ingestion.code_chunker import CodeChunker
        
        chunks = []
        base_meta = {
            "source_path": source_path,
            "source_url": source_url,
            "source_type": "github_code",
            "repo_name": repo_name,
        }
        
        # Use code-aware chunking
        chunker = CodeChunker(
            max_chunk_size=2000,
            min_chunk_size=100,
            include_imports_context=True,
        )
        
        code_chunks = chunker.chunk_file(content, source_path)
        
        for i, code_chunk in enumerate(code_chunks):
            # Create indexable text with context
            indexable_text = code_chunk.to_indexable_text()
            
            chunks.append(IngestedChunk(
                content=indexable_text,
                meta={
                    **base_meta,
                    "chunk_index": i,
                    "content_type": "code",
                    "language": code_chunk.language.value,
                    "block_type": code_chunk.block_type,
                    "block_name": code_chunk.block_name,
                    "start_line": code_chunk.start_line,
                    "end_line": code_chunk.end_line,
                },
            ))
        
        # Fallback if no chunks were created
        if not chunks:
            chunks.append(IngestedChunk(
                content=f"File: {source_path}\n\n{content}",
                meta={
                    **base_meta,
                    "chunk_index": 0,
                    "content_type": "code_file",
                },
            ))
        
        return chunks

    def _chunk_markdown_content(
        self,
        content: str,
        source_path: str,
        source_url: str,
        repo_name: str,
    ) -> List[IngestedChunk]:
        """
        Chunk markdown content intelligently.
        
        This handles:
        - Q&A style content (splits by question)
        - Regular markdown (splits by headers)
        - Tables (keeps rows together)
        
        Args:
            content: Markdown content
            source_path: Source file path
            source_url: Source URL
            repo_name: Repository name
            
        Returns:
            List of IngestedChunk objects
        """
        import re
        
        chunks = []
        base_meta = {
            "source_path": source_path,
            "source_url": source_url,
            "source_type": "github",
            "repo_name": repo_name,
        }
        
        # For Q&A content, try to detect question-answer pairs
        # Pattern: **Question text** followed by answer
        qa_pattern = re.compile(
            r'\*\*([^*]+\??)\*\*\s*\n+(.*?)(?=\*\*[^*]+\*\*\s*\n|\Z)',
            re.DOTALL
        )
        
        qa_matches = list(qa_pattern.finditer(content))
        if len(qa_matches) >= 2:
            # This looks like Q&A content
            for i, match in enumerate(qa_matches):
                question = match.group(1).strip()
                answer = match.group(2).strip()
                
                if len(answer) < 20:  # Skip too-short answers
                    continue
                
                qa_text = f"**Question:** {question}\n\n**Answer:** {answer}"
                chunks.append(IngestedChunk(
                    content=qa_text,
                    meta={
                        **base_meta,
                        "chunk_index": i,
                        "content_type": "qa",
                        "question": question,
                    },
                ))
            
            if chunks:
                return chunks
        
        # Try header-based chunking for markdown
        # Split by ## headers (keep # for title context)
        header_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        
        sections = []
        last_end = 0
        current_header = ""
        
        for match in header_pattern.finditer(content):
            # Save previous section
            if last_end > 0:
                section_content = content[last_end:match.start()].strip()
                if section_content and len(section_content) > 50:
                    full_section = f"{current_header}\n\n{section_content}" if current_header else section_content
                    sections.append(full_section)
            
            current_header = match.group(0)
            last_end = match.end()
        
        # Add final section
        if last_end < len(content):
            final = content[last_end:].strip()
            if final and len(final) > 50:
                full_section = f"{current_header}\n\n{final}" if current_header else final
                sections.append(full_section)
        
        if sections:
            for i, section in enumerate(sections):
                chunks.append(IngestedChunk(
                    content=section,
                    meta={
                        **base_meta,
                        "chunk_index": i,
                        "content_type": "section",
                    },
                ))
            return chunks
        
        # Fallback: split by paragraphs
        paragraphs = content.split("\n\n")
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) < 2000:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk and len(current_chunk) > 50:
                    chunks.append(IngestedChunk(
                        content=current_chunk,
                        meta={
                            **base_meta,
                            "chunk_index": chunk_index,
                            "content_type": "paragraph",
                        },
                    ))
                    chunk_index += 1
                current_chunk = para
        
        # Add final chunk
        if current_chunk and len(current_chunk) > 50:
            chunks.append(IngestedChunk(
                content=current_chunk,
                meta={
                    **base_meta,
                    "chunk_index": chunk_index,
                    "content_type": "paragraph",
                },
            ))
        
        return chunks

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        show_result: bool = True,
        show_metrics: bool = True,
        retrieval_mode: str = "hybrid",
        save_path: Optional[str] = None,
        compact: bool = False,
    ) -> PipelineResult:
        """
        Query the RAG system.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for history
            show_result: Whether to display result
            show_metrics: Whether to display metrics
            retrieval_mode: Retrieval mode - "hybrid", "dense", or "bm25"
            save_path: Optional path to save report
            compact: Use compact display format
            
        Returns:
            Pipeline result
        """
        # Start or resume conversation
        if self._conversation_manager:
            if not conversation_id:
                conversation_id = self._conversation_manager.start_conversation()
            else:
                self._conversation_manager.load_conversation(conversation_id)

        # Run pipeline
        result = self._orchestrator.run(
            query=query,
            conversation_id=conversation_id,
            retrieval_mode=retrieval_mode,
        )

        # Collect metrics
        if self._config.metrics.enabled:
            self._metrics_collector.record(result.metrics)

        # Display result using professional report
        if show_result:
            print_report(
                result,
                retrieval_mode=retrieval_mode,
                show_metrics=show_metrics,
                save_path=save_path,
                compact=compact,
            )
        elif save_path:
            # Save without display
            report = QueryReport.from_pipeline_result(result, retrieval_mode)
            save_report(report, save_path)

        return result

    def query_raw(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        retrieval_mode: str = "hybrid",
    ) -> PipelineResult:
        """
        Query the RAG system and return raw result without display.
        
        Used by the TUI to get results for custom rendering.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for history
            retrieval_mode: Retrieval mode - "hybrid", "dense", or "bm25"
            
        Returns:
            Pipeline result
        """
        # Start or resume conversation
        if self._conversation_manager:
            if not conversation_id:
                conversation_id = self._conversation_manager.start_conversation()
            else:
                self._conversation_manager.load_conversation(conversation_id)

        # Run pipeline
        result = self._orchestrator.run(
            query=query,
            conversation_id=conversation_id,
            retrieval_mode=retrieval_mode,
        )

        # Collect metrics
        if self._config.metrics.enabled:
            self._metrics_collector.record(result.metrics)

        return result

    def simple_query(self, query: str, top_k: int = 5) -> str:
        """
        Execute a simple query without the full pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        simple = SimplifiedOrchestrator(
            llm=self._llm_clients.chat,
            local=self._llm_clients.local,
            store=self._store,
            config=self._config,
        )
        return simple.run(query=query, top_k=top_k)

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        show_results: bool = True,
        save_path: Optional[str] = None,
    ) -> List[Tuple["StoredDoc", float]]:
        """
        Search documents without LLM generation.
        
        Pure retrieval for quick document lookup.
        
        Args:
            query: Search query
            mode: Retrieval mode - "hybrid", "dense", or "bm25"
            top_k: Number of results to return
            show_results: Whether to display results
            save_path: Optional path to save search report
            
        Returns:
            List of (StoredDoc, score) tuples
        """
        from radiant_rag_mcp.agents import DenseRetrievalAgent, BM25RetrievalAgent, RRFAgent
        
        results: List[Tuple["StoredDoc", float]] = []
        dense_results: List[Tuple["StoredDoc", float]] = []
        bm25_results: List[Tuple["StoredDoc", float]] = []
        
        if mode in ("hybrid", "dense"):
            # Dense retrieval
            dense_agent = DenseRetrievalAgent(
                self._store, 
                self._llm_clients.local, 
                self._config.retrieval
            )
            dense_result = dense_agent.run(query=query, top_k=top_k)
            dense_results = dense_result.data if hasattr(dense_result, 'data') else dense_result
            if mode == "dense":
                results = dense_results
        
        if mode in ("hybrid", "bm25"):
            # BM25 retrieval
            bm25_agent = BM25RetrievalAgent(
                self._bm25_index, 
                self._config.retrieval
            )
            bm25_result = bm25_agent.run(query=query, top_k=top_k)
            bm25_results = bm25_result.data if hasattr(bm25_result, 'data') else bm25_result
            if mode == "bm25":
                results = bm25_results
        
        # Hybrid fusion
        if mode == "hybrid":
            if dense_results and bm25_results:
                rrf_agent = RRFAgent(self._config.retrieval)
                rrf_result = rrf_agent.run(runs=[dense_results, bm25_results], top_k=top_k)
                results = (rrf_result.data if hasattr(rrf_result, 'data') else rrf_result)[:top_k]
            else:
                results = dense_results if dense_results else bm25_results
        
        # Display results using professional format
        if show_results:
            display_search_results(query, results[:top_k], mode)
        
        # Save report if requested
        if save_path:
            saved = save_search_report(query, results[:top_k], save_path, mode)
            if show_results:
                console.print(f"[dim]Report saved to: {saved}[/dim]")
        
        return results

    def start_conversation(
        self,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Start a new conversation.
        
        Args:
            conversation_id: Optional ID (generated if not provided)
            
        Returns:
            Conversation ID
        """
        if not self._conversation_manager:
            return conversation_id or str(uuid.uuid4())
        
        return self._conversation_manager.start_conversation(conversation_id)

    def get_conversation_history(
        self,
        conversation_id: str,
        max_turns: int = 10,
    ) -> str:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            max_turns: Maximum turns to return
            
        Returns:
            Formatted history string
        """
        if not self._conversation_manager:
            return ""
        
        if self._conversation_manager.load_conversation(conversation_id):
            return self._conversation_manager.get_history_for_synthesis(max_turns)
        
        return ""

    def rebuild_bm25_index(self, limit: int = 0) -> int:
        """
        Rebuild BM25 index from all documents in store.
        
        Args:
            limit: Maximum documents to index (0 = use config)
            
        Returns:
            Number of documents indexed
        """
        return self._bm25_index.build_from_store(limit=limit)

    def clear_index(self, keep_bm25: bool = False) -> bool:
        """
        Clear all documents from the index.
        
        Args:
            keep_bm25: If True, keep BM25 index (only clear vector store)
            
        Returns:
            True if successful
        """
        try:
            # Drop the Redis vector index and documents
            self._store.drop_index(delete_documents=True)
            
            # Recreate empty index with correct embedding dimension
            embedding_dim = self._llm_clients.local.embedding_dim
            self._store._ensure_index(embedding_dim)
            
            # Clear BM25 index unless told to keep it
            if not keep_bm25:
                self._bm25_index.clear()
                self._bm25_index.save()
            
            logger.info("Cleared all documents from index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "health": self.check_health(),
            "metrics": self._metrics_collector.to_dict() if self._config.metrics.enabled else None,
        }

    def display_stats(self) -> None:
        """Display system statistics."""
        health = self.check_health()
        display_index_stats(
            health.get("vector_index", {}),
            health.get("bm25_index", {}),
        )


def create_app(config_path: Optional[str] = None) -> RadiantRAG:
    """
    Create a RadiantRAG application instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        RadiantRAG instance
    """
    return RadiantRAG(config_path=config_path)


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Radiant Agentic RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to ingest",
    )
    ingest_parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat storage instead of hierarchical",
    )
    ingest_parser.add_argument(
        "--url", "-u",
        action="append",
        dest="urls",
        metavar="URL",
        help="URL to ingest (can be specified multiple times)",
    )
    ingest_parser.add_argument(
        "--crawl-depth",
        type=int,
        default=None,
        metavar="N",
        help="Crawl depth for URLs (overrides config, 0=no crawling)",
    )
    ingest_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Maximum pages to crawl (overrides config)",
    )
    ingest_parser.add_argument(
        "--no-crawl",
        action="store_true",
        help="Disable crawling, only fetch specified URLs",
    )
    ingest_parser.add_argument(
        "--auth",
        type=str,
        metavar="USER:PASS",
        help="Basic auth credentials for URL ingestion",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument(
        "query",
        type=str,
        help="Query string",
    )
    query_parser.add_argument(
        "--conversation", "-conv",
        type=str,
        help="Conversation ID for history",
    )
    query_parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simplified pipeline",
    )
    query_parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode: hybrid (default), dense (embedding only), bm25 (keyword only)",
    )
    query_parser.add_argument(
        "--save", "-s",
        type=str,
        metavar="PATH",
        help="Save report to file (supports .md, .html, .json)",
    )
    query_parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact display format",
    )

    # Search command (retrieval only, no LLM generation)
    search_parser = subparsers.add_parser("search", help="Search documents (retrieval only, no LLM)")
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode: hybrid (default), dense, or bm25",
    )
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    search_parser.add_argument(
        "--save", "-s",
        type=str,
        metavar="PATH",
        help="Save results to file (supports .md, .json)",
    )

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    interactive_parser.add_argument(
        "--tui",
        action="store_true",
        help="Use Textual-based terminal UI (default: classic mode)",
    )
    interactive_parser.add_argument(
        "--classic",
        action="store_true",
        help="Use classic command-line interactive mode",
    )

    # Stats command
    subparsers.add_parser("stats", help="Display system statistics")

    # Health command
    subparsers.add_parser("health", help="Check system health")

    # Rebuild BM25 command
    rebuild_parser = subparsers.add_parser("rebuild-bm25", help="Rebuild BM25 index")
    rebuild_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum documents to index",
    )

    # Clear index command
    clear_parser = subparsers.add_parser("clear", help="Clear all indexed documents")
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm clearing without prompt",
    )
    clear_parser.add_argument(
        "--keep-bm25",
        action="store_true",
        help="Keep BM25 index (only clear vector store)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        app = create_app(args.config)

        if args.command == "ingest":
            # Check if we have any sources to ingest
            paths = args.paths or []
            urls = args.urls or []
            
            if not paths and not urls:
                display_error("No paths or URLs specified for ingestion")
                return 1
            
            total_stats = {
                "files_processed": 0,
                "files_failed": 0,
                "chunks_created": 0,
                "documents_stored": 0,
                "urls_crawled": 0,
                "errors": [],
            }
            
            # Process local files
            if paths:
                stats = app.ingest_documents(
                    paths,
                    use_hierarchical=not args.flat,
                )
                total_stats["files_processed"] += stats.get("files_processed", 0)
                total_stats["files_failed"] += stats.get("files_failed", 0)
                total_stats["chunks_created"] += stats.get("chunks_created", 0)
                total_stats["documents_stored"] += stats.get("documents_stored", 0)
                total_stats["errors"].extend(stats.get("errors", []))
            
            # Process URLs
            if urls:
                # Parse auth if provided
                basic_auth = None
                if args.auth:
                    if ":" in args.auth:
                        basic_auth = tuple(args.auth.split(":", 1))
                    else:
                        display_error("Auth must be in format USER:PASSWORD")
                        return 1
                
                # Determine crawl depth
                crawl_depth = args.crawl_depth
                if args.no_crawl:
                    crawl_depth = 0
                
                url_stats = app.ingest_urls(
                    urls,
                    use_hierarchical=not args.flat,
                    crawl_depth=crawl_depth,
                    max_pages=args.max_pages,
                    basic_auth=basic_auth,
                )
                total_stats["urls_crawled"] += url_stats.get("urls_crawled", 0)
                total_stats["files_processed"] += url_stats.get("files_processed", 0)
                total_stats["documents_stored"] += url_stats.get("documents_stored", 0)
                total_stats["errors"].extend(url_stats.get("errors", []))
            
            # Display results
            summary_parts = []
            if total_stats["files_processed"]:
                summary_parts.append(f"{total_stats['files_processed']} files")
            if total_stats["urls_crawled"]:
                summary_parts.append(f"{total_stats['urls_crawled']} URLs")
            
            display_success(
                f"Processed {', '.join(summary_parts)}, "
                f"created {total_stats['documents_stored']} documents",
                title="Ingestion Complete",
            )
            if total_stats["errors"]:
                for error in total_stats["errors"][:5]:
                    console.print(f"[yellow]Warning: {error}[/]")

        elif args.command == "query":
            if args.simple:
                answer = app.simple_query(args.query)
                console.print(answer)
            else:
                app.query(
                    args.query,
                    conversation_id=args.conversation,
                    retrieval_mode=args.mode,
                    save_path=args.save,
                    compact=args.compact,
                )
                if args.save:
                    console.print("[green]✓[/green] Report saved to: {args.save}".format(args=args))

        elif args.command == "search":
            app.search(
                args.query,
                mode=args.mode,
                top_k=args.top_k,
                show_results=True,
                save_path=args.save,
            )

        elif args.command == "interactive":
            if args.tui and not args.classic:
                # Use Textual TUI
                try:
                    from radiant_rag_mcp.ui.tui import run_tui
                    run_tui(app)
                except ImportError as e:
                    display_error(
                        f"Textual TUI not available: {e}. "
                        "Install with: pip install textual>=0.40.0"
                    )
                    display_info("Falling back to classic mode...")
                    run_interactive(app)
            else:
                # Use classic interactive mode
                run_interactive(app)

        elif args.command == "stats":
            app.display_stats()

        elif args.command == "health":
            health = app.check_health()
            if health["redis"]:
                display_success("Redis: Connected")
            else:
                display_error("Redis: Not connected")
            
            display_info(f"Vector index: {health.get('vector_index', {})}")
            display_info(f"BM25 index: {health.get('bm25_index', {})}")

        elif args.command == "rebuild-bm25":
            count = app.rebuild_bm25_index(args.limit)
            display_success(f"Rebuilt BM25 index with {count} documents")

        elif args.command == "clear":
            # Get current document count
            doc_count = app.store.count_documents()
            
            if doc_count == 0:
                display_info("Index is already empty")
                return 0
            
            # Confirm unless --confirm flag is set
            if not args.confirm:
                console.print(f"[yellow]Warning: This will delete {doc_count} documents from the index.[/yellow]")
                response = console.input("[bold]Are you sure? Type 'yes' to confirm: [/bold]")
                if response.lower() != "yes":
                    display_info("Cancelled")
                    return 0
            
            # Clear the index
            cleared = app.clear_index(keep_bm25=args.keep_bm25)
            if cleared:
                display_success(f"Cleared {doc_count} documents from the index")
            else:
                display_error("Failed to clear index")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        return 130
    except Exception as e:
        display_error(str(e), title="Fatal Error")
        logger.exception("Fatal error")
        return 1


def run_interactive(app: RadiantRAG) -> None:
    """Run interactive query mode with professional output."""
    console.print()
    console.print("[bold blue]━━━ Agentic RAG Interactive Mode ━━━[/bold blue]")
    console.print()
    console.print("[dim]Commands:[/dim]")
    console.print("  [cyan]quit[/cyan]         Exit interactive mode")
    console.print("  [cyan]new[/cyan]          Start new conversation")
    console.print("  [cyan]stats[/cyan]        Show system statistics")
    console.print("  [cyan]mode[/cyan] MODE    Set retrieval mode (hybrid/dense/bm25)")
    console.print("  [cyan]save[/cyan] PATH    Save last result to file (markdown/json)")
    console.print("  [cyan]report[/cyan] PATH  Save detailed text report")
    console.print("  [cyan]search[/cyan] Q     Search without LLM generation")
    console.print()

    conversation_id = app.start_conversation()
    console.print(f"[dim]Conversation: {conversation_id[:8]}...[/dim]")
    console.print()
    
    current_mode = "hybrid"
    last_result = None

    while True:
        try:
            # Show current mode in prompt
            mode_indicator = {"hybrid": "H", "dense": "D", "bm25": "B"}[current_mode]
            query = console.input(f"[bold green][{mode_indicator}]>[/bold green] ").strip()
            
            if not query:
                continue
            
            # Commands
            if query.lower() in ("quit", "exit", "q"):
                break
            
            if query.lower() == "stats":
                app.display_stats()
                continue
            
            if query.lower() == "new":
                conversation_id = app.start_conversation()
                console.print(f"[dim]New conversation: {conversation_id[:8]}...[/dim]")
                console.print()
                continue
            
            if query.lower().startswith("mode "):
                new_mode = query[5:].strip().lower()
                if new_mode in ("hybrid", "dense", "bm25"):
                    current_mode = new_mode
                    console.print(f"[dim]Retrieval mode: {current_mode}[/dim]")
                else:
                    console.print("[yellow]Valid modes: hybrid, dense, bm25[/yellow]")
                continue
            
            if query.lower().startswith("save "):
                if last_result:
                    save_path = query[5:].strip()
                    report = QueryReport.from_pipeline_result(last_result, current_mode)
                    saved = save_report(report, save_path)
                    console.print(f"[green]✓[/green] Saved to: {saved}")
                else:
                    console.print("[yellow]No previous result to save[/yellow]")
                continue
            
            if query.lower().startswith("report "):
                if last_result:
                    save_path = query[7:].strip()
                    try:
                        from radiant_rag_mcp.ui.reports.text import save_text_report
                        saved = save_text_report(
                            last_result,
                            save_path,
                            retrieval_mode=current_mode,
                        )
                        console.print(f"[green]✓[/green] Text report saved to: {saved}")
                    except Exception as e:
                        console.print(f"[red]Error saving report: {e}[/red]")
                else:
                    console.print("[yellow]No previous result to save[/yellow]")
                continue
            
            if query.lower().startswith("search "):
                search_query = query[7:].strip()
                if search_query:
                    app.search(search_query, mode=current_mode, top_k=10)
                continue
            
            # Regular query
            last_result = app.query(
                query, 
                conversation_id=conversation_id,
                retrieval_mode=current_mode,
            )

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit[/dim]")
        except EOFError:
            break

    console.print()
    console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    sys.exit(main())
