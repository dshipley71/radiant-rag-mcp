"""
================================================================================
AGENT TEMPLATE FOR RADIANT RAG ARCHITECTURE
================================================================================

This file provides comprehensive templates for integrating new agents into the
Radiant RAG system using the formal BaseAgent abstract base class.

All agents should inherit from BaseAgent (or its subclasses LLMAgent/RetrievalAgent)
to ensure consistent behavior, logging, metrics collection, and error handling.

================================================================================
INTEGRATION CHECKLIST
================================================================================

Follow these steps to integrate a new agent:

STEP 1: CHOOSE YOUR BASE CLASS
------------------------------
Select the appropriate base class for your agent:

- BaseAgent: General-purpose base with full lifecycle management
- LLMAgent: For agents that primarily use LLM for processing
- RetrievalAgent: For agents that perform document retrieval

STEP 2: CREATE YOUR AGENT FILE
------------------------------
Create: radiant/agents/your_agent.py

Use one of the templates below as a starting point.

STEP 3: CREATE CONFIGURATION DATACLASS
--------------------------------------
Add to radiant/config.py:

```python
@dataclass(frozen=True)
class YourAgentConfig:
    '''Configuration for YourAgent.'''
    enabled: bool = True
    # Add your config parameters here
    param1: int = 10
    param2: float = 0.5
    param3: str = "default"
```

STEP 4: UPDATE AppConfig
------------------------
Add to the AppConfig dataclass in radiant/config.py:

```python
@dataclass(frozen=True)
class AppConfig:
    # ... existing fields ...
    your_agent: YourAgentConfig = field(default_factory=YourAgentConfig)
```

STEP 5: UPDATE load_config() FUNCTION
-------------------------------------
Add parsing logic in radiant/config.py load_config():

```python
your_agent = YourAgentConfig(
    enabled=_get_config_value(data, "your_agent", "enabled", True, _parse_bool),
    param1=_get_config_value(data, "your_agent", "param1", 10, _parse_int),
    param2=_get_config_value(data, "your_agent", "param2", 0.5, _parse_float),
    param3=_get_config_value(data, "your_agent", "param3", "default"),
)
```

Don't forget to add `your_agent=your_agent` to the AppConfig constructor call.

STEP 6: ADD YAML CONFIGURATION
------------------------------
Add to config.yaml:

```yaml
your_agent:
  enabled: true
  param1: 10
  param2: 0.5
  param3: default
```

STEP 7: UPDATE AGENTS __init__.py
---------------------------------
Export your agent in radiant/agents/__init__.py:

```python
from radiant_rag_mcp.agents.your_agent import (
    YourAgent,
    YourAgentResult,  # if you have result dataclasses
)

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "YourAgent",
    "YourAgentResult",
]
```

STEP 8: INTEGRATE INTO ORCHESTRATOR
-----------------------------------
In radiant/orchestrator.py:

a) Import your agent:
   ```python
   from radiant_rag_mcp.agents import YourAgent, YourAgentResult
   ```

b) Add instance variable in RAGOrchestrator.__init__:
   ```python
   self._your_agent: Optional[YourAgent] = None
   if config.your_agent.enabled:
       self._your_agent = YourAgent(
           llm=llm,
           param1=config.your_agent.param1,
           param2=config.your_agent.param2,
       )
   ```

c) Create a runner method:
   ```python
   def _run_your_agent(
       self,
       ctx: AgentContext,
       metrics: RunMetrics,
   ) -> Optional[YourAgentResult]:
       '''Execute your agent.'''
       if not self._your_agent:
           return None
       
       with metrics.track_step("YourAgent") as step:
           result = self._your_agent.run(
               correlation_id=ctx.run_id,
               query=ctx.original_query,
           )
           
           if result.success:
               step.extra["confidence"] = result.metrics.confidence
               step.extra["items"] = result.metrics.items_returned
           else:
               metrics.mark_degraded("your_agent", result.error)
           
           return result
   ```

d) Call from the main run() method at the appropriate pipeline phase

STEP 9: UPDATE AgentContext IF NEEDED
-------------------------------------
If your agent produces data that should persist through the pipeline,
add fields to AgentContext in radiant/agents/base.py:

```python
@dataclass
class AgentContext:
    # ... existing fields ...
    your_agent_results: List[Any] = field(default_factory=list)
    your_agent_score: float = 0.0
```

STEP 10: UPDATE PipelineResult IF NEEDED
----------------------------------------
If your agent's results should be in the final output,
add fields to PipelineResult in radiant/orchestrator.py:

```python
@dataclass
class PipelineResult:
    # ... existing fields ...
    your_agent_used: bool = False
    your_agent_data: Optional[Dict[str, Any]] = None
```

STEP 11: WRITE TESTS
--------------------
Create test file: tests/test_agents/test_your_agent.py

```python
import unittest
from unittest.mock import MagicMock, Mock

class TestYourAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.agent = YourAgent(llm=self.mock_llm)
    
    def test_run_success(self):
        # Test implementation
        pass
    
    def test_run_failure_handling(self):
        # Test error handling
        pass
```

================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Import the formal base classes
from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
    LLMAgent,
    RetrievalAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient, LocalNLPModels
    from radiant_rag_mcp.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class CustomAgentData:
    """
    Custom data structure for your agent's output.
    
    Define all output fields your agent produces.
    """
    
    output: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output,
            "items": self.items,
            "score": self.score,
            "metadata": self.metadata,
        }


# =============================================================================
# TEMPLATE 1: LLM-BASED AGENT (Inherits from LLMAgent)
# =============================================================================

class TemplateProcessingAgent(LLMAgent):
    """
    Template for LLM-based processing agents.
    
    Use this pattern when your agent primarily uses LLM for:
    - Query analysis and planning
    - Text transformation and rewriting
    - Classification and categorization
    - Answer synthesis and critique
    
    Example usage:
        agent = TemplateProcessingAgent(
            llm=llm_client,
            param1=10,
            param2=0.5,
        )
        result = agent.run(query="What is machine learning?")
        
        if result.success:
            print(f"Output: {result.data.output}")
            print(f"Confidence: {result.metrics.confidence}")
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        param1: int = 10,
        param2: float = 0.5,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the processing agent.
        
        Args:
            llm: LLM client for reasoning (required)
            param1: Example integer parameter
            param2: Example float parameter
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)
        self._param1 = param1
        self._param2 = param2
    
    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "TemplateProcessingAgent"
    
    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.QUERY_PROCESSING
    
    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Template agent for LLM-based query processing"
    
    def _execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> CustomAgentData:
        """
        Execute the agent's core processing logic.
        
        Args:
            query: User query to process
            context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            CustomAgentData with processing results
        """
        # Build prompts
        system = self._build_system_prompt(context)
        user = self._build_user_prompt(query, kwargs)
        
        # Call LLM for JSON response
        result = self._chat_json(
            system=system,
            user=user,
            default={"output": "", "score": 0.0, "items": []},
            expected_type=dict,
        )
        
        # Process and validate response
        output = str(result.get("output", ""))
        score = float(result.get("score", 0.0))
        items = result.get("items", [])
        
        if not isinstance(items, list):
            items = []
        
        return CustomAgentData(
            output=output,
            items=items,
            score=max(0.0, min(1.0, score)),
            metadata={"param1": self._param1, "param2": self._param2},
        )
    
    def _build_system_prompt(self, context: Optional[str] = None) -> str:
        """Build the system prompt for the LLM."""
        prompt = """You are a specialized processing agent in a RAG pipeline.
Your task is to analyze the input and produce structured output.

Guidelines:
1. Be concise and accurate
2. Follow the output format strictly
3. Provide confidence scores

Return a JSON object:
{
    "output": "your processed output",
    "score": 0.0-1.0,
    "items": [{"key": "value"}]
}"""
        
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        
        return prompt
    
    def _build_user_prompt(
        self,
        query: str,
        kwargs: Dict[str, Any],
    ) -> str:
        """Build the user prompt for the LLM."""
        parts = [f"Input: {query}"]
        
        if kwargs.get("hints"):
            parts.append(f"Hints: {kwargs['hints']}")
        
        parts.append("\nReturn JSON only.")
        return "\n".join(parts)
    
    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[CustomAgentData]:
        """
        Handle errors with fallback behavior.
        
        Override to provide custom error recovery.
        """
        self.logger.warning(f"Error occurred, providing fallback: {error}")
        
        # Return a fallback result
        return CustomAgentData(
            output="Processing failed, using fallback",
            items=[],
            score=0.0,
            metadata={"error": str(error), "fallback": True},
        )


# =============================================================================
# TEMPLATE 2: RETRIEVAL-BASED AGENT (Inherits from RetrievalAgent)
# =============================================================================

class TemplateRetrievalAgent(RetrievalAgent):
    """
    Template for retrieval-based agents.
    
    Use this pattern when your agent performs:
    - Vector similarity search
    - Document retrieval and ranking
    - Semantic search operations
    
    Example usage:
        agent = TemplateRetrievalAgent(
            store=vector_store,
            local_models=local_models,
            top_k=10,
        )
        result = agent.run(query="machine learning basics")
        
        if result.success:
            for doc, score in result.data:
                print(f"Score: {score:.3f} - {doc.content[:100]}")
    """
    
    def __init__(
        self,
        store: "BaseVectorStore",
        local_models: "LocalNLPModels",
        top_k: int = 10,
        min_similarity: float = 0.0,
        search_scope: str = "leaves",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the retrieval agent.
        
        Args:
            store: Vector store for retrieval
            local_models: Local NLP models for embedding
            top_k: Maximum documents to retrieve
            min_similarity: Minimum similarity threshold
            search_scope: Document scope ("leaves", "parents", "all")
            enabled: Whether the agent is enabled
        """
        super().__init__(store=store, local_models=local_models, enabled=enabled)
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._search_scope = search_scope
    
    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "TemplateRetrievalAgent"
    
    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.RETRIEVAL
    
    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Template agent for semantic document retrieval"
    
    def _execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[tuple]:
        """
        Execute document retrieval.
        
        Args:
            query: Search query
            top_k: Override for number of results
            filters: Optional metadata filters
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._top_k
        
        # Generate query embedding
        query_embedding = self._embed(query)
        
        # Apply any preprocessing
        query_embedding = self._preprocess_embedding(query_embedding)
        
        # Determine document level filter
        doc_level_filter = self._get_doc_level_filter()
        
        # Retrieve documents
        results = self._retrieve(
            query_embedding=query_embedding,
            top_k=k,
            min_similarity=self._min_similarity,
            doc_level_filter=doc_level_filter,
        )
        
        # Apply post-processing filters
        if filters:
            results = self._apply_filters(results, filters)
        
        self.logger.debug(
            f"Retrieved {len(results)} documents",
            query_length=len(query),
            top_k=k,
        )
        
        return results
    
    def _get_doc_level_filter(self) -> Optional[str]:
        """Convert search scope to doc level filter."""
        scope_map = {
            "leaves": "child",
            "parents": "parent",
            "all": None,
        }
        return scope_map.get(self._search_scope, "child")
    
    def _preprocess_embedding(
        self,
        embedding: List[float],
    ) -> List[float]:
        """
        Preprocess the query embedding.
        
        Override to add normalization, dimensionality reduction, etc.
        """
        return embedding
    
    def _apply_filters(
        self,
        results: List[tuple],
        filters: Dict[str, Any],
    ) -> List[tuple]:
        """
        Apply metadata filters to results.
        
        Override to implement custom filtering logic.
        """
        filtered = []
        for doc, score in results:
            if self._matches_filters(doc, filters):
                filtered.append((doc, score))
        return filtered
    
    def _matches_filters(
        self,
        doc: Any,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if document matches filters."""
        meta = getattr(doc, 'meta', {}) or {}
        
        for key, value in filters.items():
            if key not in meta:
                return False
            if meta[key] != value:
                return False
        
        return True


# =============================================================================
# TEMPLATE 3: COMPLEX MULTI-STEP AGENT (Inherits from BaseAgent)
# =============================================================================

@dataclass
class MultiStepResult:
    """Result from multi-step processing."""
    
    final_output: str
    steps: List[Dict[str, Any]]
    total_steps: int
    accumulated_context: List[Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_output": self.final_output,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "context_size": len(self.accumulated_context),
        }


class TemplateMultiStepAgent(BaseAgent):
    """
    Template for complex multi-step agents.
    
    Use this pattern when your agent:
    - Combines LLM reasoning with retrieval
    - Performs iterative processing
    - Requires multiple passes to complete
    
    Example usage:
        agent = TemplateMultiStepAgent(
            llm=llm_client,
            store=vector_store,
            local_models=local_models,
            max_steps=5,
        )
        result = agent.run(query="complex multi-part question")
        
        if result.success:
            print(f"Completed in {result.data.total_steps} steps")
            print(f"Output: {result.data.final_output}")
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        store: "BaseVectorStore",
        local_models: "LocalNLPModels",
        max_steps: int = 5,
        confidence_threshold: float = 0.7,
        docs_per_step: int = 3,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the multi-step agent.
        
        Args:
            llm: LLM client for reasoning
            store: Vector store for retrieval
            local_models: Local models for embedding
            max_steps: Maximum processing steps
            confidence_threshold: Confidence to stop early
            docs_per_step: Documents to retrieve per step
            enabled: Whether the agent is enabled
        """
        super().__init__(
            llm=llm,
            store=store,
            local_models=local_models,
            enabled=enabled,
        )
        self._max_steps = max_steps
        self._confidence_threshold = confidence_threshold
        self._docs_per_step = docs_per_step
    
    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "TemplateMultiStepAgent"
    
    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.EVALUATION
    
    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Template agent for iterative multi-step processing"
    
    def _execute(
        self,
        query: str,
        initial_context: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> MultiStepResult:
        """
        Execute multi-step processing.
        
        Args:
            query: User query
            initial_context: Optional starting context
            **kwargs: Additional arguments
            
        Returns:
            MultiStepResult with processing results
        """
        steps = []
        accumulated_context = list(initial_context or [])
        accumulated_knowledge = ""
        
        for step_num in range(self._max_steps):
            self.logger.debug(f"Starting step {step_num + 1}")
            
            # Generate query for this step
            step_query = self._generate_step_query(
                query,
                accumulated_knowledge,
                step_num,
            )
            
            # Retrieve relevant documents
            new_docs = self._retrieve_for_step(step_query)
            accumulated_context.extend(new_docs)
            
            # Extract information from documents
            extracted, confidence = self._extract_information(
                step_query,
                new_docs,
                accumulated_knowledge,
            )
            
            # Record step
            steps.append({
                "step": step_num + 1,
                "query": step_query,
                "docs_retrieved": len(new_docs),
                "extracted": extracted,
                "confidence": confidence,
            })
            
            # Accumulate knowledge
            if extracted:
                accumulated_knowledge += f" {extracted}"
            
            # Check if we should stop
            if confidence >= self._confidence_threshold:
                self.logger.info(
                    f"Reached confidence threshold at step {step_num + 1}"
                )
                break
        
        # Synthesize final output
        final_output = self._synthesize_output(
            query,
            accumulated_knowledge,
            steps,
        )
        
        return MultiStepResult(
            final_output=final_output,
            steps=steps,
            total_steps=len(steps),
            accumulated_context=accumulated_context,
        )
    
    def _generate_step_query(
        self,
        original_query: str,
        knowledge: str,
        step: int,
    ) -> str:
        """Generate query for current step."""
        if step == 0 or not knowledge:
            return original_query
        
        # Use LLM to generate follow-up query
        system = """Generate a follow-up query based on what we've learned.
Return JSON: {"query": "the follow-up query"}"""
        
        user = f"""Original: {original_query}
Learned: {knowledge}
Step: {step + 1}

Return JSON only."""
        
        result, _ = self._llm.chat_json(
            system=system,
            user=user,
            default={"query": original_query},
            expected_type=dict,
        )
        
        return str(result.get("query", original_query))
    
    def _retrieve_for_step(self, query: str) -> List[Any]:
        """Retrieve documents for current step."""
        try:
            query_embedding = self._local_models.embed_single(query)
            results = self._store.retrieve_by_embedding(
                query_embedding,
                top_k=self._docs_per_step,
            )
            return [doc for doc, _ in results]
        except Exception as e:
            self.logger.warning(f"Retrieval failed: {e}")
            return []
    
    def _extract_information(
        self,
        query: str,
        docs: List[Any],
        prior_knowledge: str,
    ) -> tuple:
        """Extract information from documents."""
        if not docs:
            return "", 0.0
        
        # Format context
        context = "\n\n".join([
            f"[{i}] {getattr(doc, 'content', str(doc))[:1000]}"
            for i, doc in enumerate(docs[:5], 1)
        ])
        
        system = """Extract relevant information from context.
Return JSON: {"extracted": "information", "confidence": 0.0-1.0}"""
        
        user = f"""Query: {query}
Prior: {prior_knowledge}

Context:
{context}

Return JSON only."""
        
        result, _ = self._llm.chat_json(
            system=system,
            user=user,
            default={"extracted": "", "confidence": 0.0},
            expected_type=dict,
        )
        
        return (
            str(result.get("extracted", "")),
            float(result.get("confidence", 0.0)),
        )
    
    def _synthesize_output(
        self,
        query: str,
        knowledge: str,
        steps: List[Dict[str, Any]],
    ) -> str:
        """Synthesize final output from accumulated knowledge."""
        system = "Synthesize a comprehensive answer from the accumulated knowledge."
        
        user = f"""Query: {query}

Knowledge: {knowledge}

Steps completed: {len(steps)}

Provide a clear, comprehensive answer."""
        
        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if not response.success:
            return knowledge
        
        return response.content.strip()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_documents_for_llm(
    docs: List[Any],
    max_docs: int = 10,
    max_chars: int = 2000,
) -> str:
    """
    Format documents as context string for LLM.
    
    Args:
        docs: List of documents
        max_docs: Maximum documents to include
        max_chars: Maximum characters per document
        
    Returns:
        Formatted context string
    """
    parts = []
    
    for i, doc in enumerate(docs[:max_docs], 1):
        content = getattr(doc, 'content', str(doc))
        
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        source = ""
        if hasattr(doc, 'meta') and doc.meta:
            source = doc.meta.get("source_path", "")
            if source:
                source = f" (Source: {source})"
        
        parts.append(f"[DOC {i}]{source}\n{content}")
    
    return "\n\n".join(parts)


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if "enabled" not in config:
        raise ValueError("Config must have 'enabled' field")
    
    return True


# =============================================================================
# EXAMPLE: REGISTERING WITH GLOBAL REGISTRY
# =============================================================================

def register_custom_agents(
    llm: "LLMClient",
    store: Optional["BaseVectorStore"] = None,
    local_models: Optional["LocalNLPModels"] = None,
) -> None:
    """
    Register custom agents with the global registry.
    
    Call this during application initialization to make
    agents discoverable via the registry.
    
    Args:
        llm: LLM client
        store: Optional vector store
        local_models: Optional local models
    """
    from radiant_rag_mcp.agents.registry import get_global_registry
    
    registry = get_global_registry()
    
    # Register processing agent
    processing_agent = TemplateProcessingAgent(llm=llm)
    registry.register_instance(
        instance=processing_agent,
        name=processing_agent.name,
        description=processing_agent.description,
        category=processing_agent.category.value,
        method_name="run",
    )
    
    # Register retrieval agent if dependencies available
    if store and local_models:
        retrieval_agent = TemplateRetrievalAgent(
            store=store,
            local_models=local_models,
        )
        registry.register_instance(
            instance=retrieval_agent,
            name=retrieval_agent.name,
            description=retrieval_agent.description,
            category=retrieval_agent.category.value,
            method_name="run",
        )
    
    logger.info(f"Registered {len(registry)} agents with global registry")


# =============================================================================
# MODULE INFO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RADIANT RAG AGENT TEMPLATE")
    print("=" * 70)
    print()
    print("This module provides templates for creating new agents that inherit")
    print("from the formal BaseAgent abstract base class.")
    print()
    print("Available base classes:")
    print("  - BaseAgent: General-purpose with full lifecycle management")
    print("  - LLMAgent: For LLM-driven processing agents")
    print("  - RetrievalAgent: For document retrieval agents")
    print()
    print("Template agents included:")
    print("  - TemplateProcessingAgent (LLMAgent)")
    print("  - TemplateRetrievalAgent (RetrievalAgent)")
    print("  - TemplateMultiStepAgent (BaseAgent)")
    print()
    print("See the module docstring for the complete integration checklist.")
    print("=" * 70)
