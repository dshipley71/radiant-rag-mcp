# AGENTS.md — Radiant RAG MCP Agents

Developer guide for creating and modifying pipeline agents in `radiant-rag-mcp`.

Package: `radiant_rag_mcp`  
Source layout: `src/radiant_rag_mcp/`

---

## Agent Hierarchy

```
BaseAgent (Abstract)
├── LLMAgent (requires LLM client)
│   ├── PlanningAgent               (disabled by default)
│   ├── QueryDecompositionAgent     (disabled by default)
│   ├── QueryRewriteAgent
│   ├── QueryExpansionAgent
│   ├── AnswerSynthesisAgent
│   ├── CriticAgent
│   ├── ContextEvaluationAgent
│   ├── SummarizationAgent
│   ├── FactVerificationAgent       (disabled by default)
│   ├── CitationTrackingAgent
│   ├── LanguageDetectionAgent      (disabled by default)
│   ├── TranslationAgent            (disabled by default)
│   ├── IntelligentChunkingAgent
│   └── WebSearchAgent              (disabled by default)
│
├── RetrievalAgent (requires vector store)
│   └── DenseRetrievalAgent
│
└── BaseAgent (direct inheritance)
    ├── BM25RetrievalAgent
    ├── RRFAgent
    ├── HierarchicalAutoMergingAgent
    ├── CrossEncoderRerankingAgent
    └── MultiHopReasoningAgent      (disabled by default)
```

---

## Creating a New Agent

### Step 1: Choose base class

```python
from radiant_rag_mcp.agents.base_agent import BaseAgent, LLMAgent, RetrievalAgent, AgentCategory
```

- **`BaseAgent`** — no LLM or vector store required (e.g. BM25, RRF, reranking)
- **`LLMAgent`** — requires an `LLMClient` for text generation
- **`RetrievalAgent`** — requires a `BaseVectorStore` for document retrieval

### Step 2: Implement required properties

```python
class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MyAgent"

    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY  # see AgentCategory enum below

    @property
    def description(self) -> str:
        return "What this agent does"
```

### Step 3: Implement `_execute()`

```python
def _execute(
    self,
    query: str,
    context: List[Any],
    **kwargs: Any,
) -> YourReturnType:
    """
    Core execution logic.

    Args:
        query: The input query
        context: Retrieved documents

    Returns:
        Your result type
    """
    # implementation here
    return result
```

### Step 4: Optional lifecycle hooks

```python
def _before_execute(self, **kwargs) -> None:
    """Called before _execute(). Use for validation or setup."""
    pass

def _after_execute(self, result: Any, metrics: AgentMetrics, **kwargs) -> Any:
    """Called after success. Can modify result or add metrics."""
    metrics.custom["my_metric"] = some_value
    return result

def _on_error(self, error: Exception, metrics: AgentMetrics, **kwargs) -> Optional[Any]:
    """Called on error. Return a fallback value or None to re-raise."""
    if isinstance(error, RecoverableError):
        return self._default_result()
    return None  # re-raises the original error
```

---

## Agent Parameter Reference

**Critical:** The orchestrator calls agents using keyword arguments.
Parameter names in `_execute()` must match exactly.

| Agent | `_execute()` Parameters |
|---|---|
| `PlanningAgent` | `query: str, context: Optional[str] = None` |
| `QueryDecompositionAgent` | `query: str` |
| `QueryRewriteAgent` | `query: str` |
| `QueryExpansionAgent` | `query: str` |
| `DenseRetrievalAgent` | `query: str, top_k: Optional[int] = None, search_scope: Optional[str] = None` |
| `BM25RetrievalAgent` | `query: str, top_k: Optional[int] = None` |
| `WebSearchAgent` | `query: str, plan: Dict[str, Any]` |
| `RRFAgent` | `runs: List[List[Tuple[Any, float]]], top_k: Optional[int] = None, rrf_k: Optional[int] = None` |
| `HierarchicalAutoMergingAgent` | `candidates: List[Tuple[Any, float]], top_k: Optional[int] = None` |
| `CrossEncoderRerankingAgent` | `query: str, docs: List[Tuple[Any, float]], top_k: Optional[int] = None` |
| `AnswerSynthesisAgent` | `query: str, docs: List[Any], conversation_history: str = ""` |
| `CriticAgent` | `query: str, answer: str, context_docs: List[Any], is_retry: bool = False, retry_count: int = 0` |
| `MultiHopReasoningAgent` | `query: str, initial_context: Optional[List[Any]] = None, force_multihop: bool = False` |
| `ContextEvaluationAgent` | `query: str, docs: List[Any]` |
| `CitationTrackingAgent` | `query: str, answer: str, docs: List[Any]` |
| `SummarizationAgent` | `docs: List[Any], query: str` |

---

## Adding an Agent to the Orchestrator

### 1. Import in `orchestrator.py`

```python
from radiant_rag_mcp.agents.my_agent import MyAgent
```

### 2. Initialize in `__init__`

```python
self._my_agent = MyAgent(llm=llm_clients.chat, config=config.my_agent)
```

### 3. Call with correct parameters

```python
result = self._my_agent.run(
    correlation_id=ctx.run_id,
    query=ctx.original_query,
    context=some_context,
)
data = _extract_agent_data(
    result,
    default=fallback_value,
    agent_name="MyAgent",
    metrics_collector=self._metrics_collector,
)
```

---

## Agent Categories

```python
class AgentCategory(Enum):
    PLANNING         = "planning"           # Query analysis, execution planning
    QUERY_PROCESSING = "query_processing"   # Decomposition, rewrite, expansion
    RETRIEVAL        = "retrieval"          # Dense, sparse, web search
    POST_RETRIEVAL   = "post_retrieval"     # Fusion, reranking, merging
    GENERATION       = "generation"         # Answer synthesis
    EVALUATION       = "evaluation"         # Critic, fact verification
    TOOL             = "tool"               # Calculator, code execution
    UTILITY          = "utility"            # General purpose
```

---

## Metrics Collection

### Automatic metrics

Every agent automatically collects:
- `duration_ms` — execution wall time
- `status` — `SUCCESS`, `FAILED`, `PARTIAL`, or `SKIPPED`
- `run_id` — unique execution ID
- `correlation_id` — request tracing ID

### Custom metrics

Add custom values in `_after_execute()`:

```python
def _after_execute(self, result, metrics, **kwargs):
    metrics.items_processed = len(result)
    metrics.confidence = self._calculate_confidence(result)
    metrics.custom["cache_hit"] = self._cache_hit
    return result
```

### Export metrics

```python
from radiant_rag_mcp.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(namespace="radiant_rag")
result = agent.run(query="test")
exporter.record_execution(result)
```

---

## Testing Agents

### Unit test template

```python
import pytest
from radiant_rag_mcp.agents.base_agent import AgentStatus

class TestMyAgent:
    def test_successful_execution(self):
        agent = MyAgent(config=my_config)
        result = agent.run(query="test")

        assert result.success is True
        assert result.status == AgentStatus.SUCCESS
        assert result.data is not None
        assert result.metrics.duration_ms > 0

    def test_handles_empty_input(self):
        agent = MyAgent(config=my_config)
        result = agent.run(query="")

        # Should degrade gracefully
        assert result.data is not None or result.success is True
```

### Run tests

```bash
pytest tests/test_base_agent_lifecycle.py -v
pytest tests/test_agents/test_my_agent.py -v
```

---

## Common Mistakes

### Wrong parameter name

```python
# ✗ WRONG — orchestrator uses 'runs' but 'retrieval_lists' is passed
result = self._rrf_agent.run(retrieval_lists=data)

# ✓ CORRECT — check _execute() signature
result = self._rrf_agent.run(runs=data)
```

### Missing correlation ID

```python
# ✗ Missing — no tracing linkage
result = agent.run(query=query)

# ✓ Include for tracing
result = agent.run(correlation_id=ctx.run_id, query=query)
```

### Using AgentResult as data directly

```python
# ✗ Wrong — ctx.plan gets an AgentResult object, not the plan dict
ctx.plan = self._planning_agent.run(query=query)

# ✓ Extract the data
result = self._planning_agent.run(query=query)
ctx.plan = result.data if result.success else default_plan
```

---

## File Reference

| File | Purpose |
|---|---|
| `base_agent.py` | `BaseAgent`, `LLMAgent`, `RetrievalAgent`, `AgentResult`, `AgentMetrics` |
| `base.py` | `AgentContext` and context utility functions |
| `agent_template.py` | Starter template for new agents |
| `registry.py` | Agent registration and discovery |
| `planning.py` | Query complexity analysis and execution planning |
| `decomposition.py` | Break complex queries into sub-queries |
| `rewrite.py` | Transform queries for better retrieval |
| `expansion.py` | Add synonyms and related terms |
| `dense.py` | Vector similarity retrieval (sentence-transformers) |
| `bm25.py` | Sparse keyword retrieval |
| `fusion.py` | Reciprocal Rank Fusion (RRF) |
| `automerge.py` | Hierarchical parent/child chunk merging |
| `rerank.py` | Cross-encoder reranking |
| `synthesis.py` | Answer generation from retrieved context |
| `critic.py` | Answer quality and confidence evaluation |
| `context_eval.py` | Pre-generation context quality gate |
| `summarization.py` | Context compression for long conversations |
| `multihop.py` | Multi-step reasoning chains |
| `fact_verification.py` | Claim verification against retrieved context |
| `citation.py` | Source attribution and audit trail generation |
| `chunking.py` | LLM-guided semantic document chunking |
| `language_detection.py` | Language identification (disabled by default) |
| `translation.py` | Cross-language document normalization (disabled by default) |
| `strategy_memory.py` | Adaptive retrieval strategy learning |
| `web_search.py` | Real-time web content augmentation (disabled by default) |
| `tools.py` | Calculator and code execution tools |
