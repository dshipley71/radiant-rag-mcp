# AGENTS.md - Radiant RAG Agents

Detailed guide for developing and modifying pipeline agents.

## Agent Hierarchy

```
BaseAgent (Abstract)
├── LLMAgent (requires LLM client)
│   ├── PlanningAgent
│   ├── AnswerSynthesisAgent
│   ├── CriticAgent
│   ├── QueryDecompositionAgent
│   ├── QueryRewriteAgent
│   ├── QueryExpansionAgent
│   └── WebSearchAgent
│
├── RetrievalAgent (requires vector store)
│   └── DenseRetrievalAgent
│
└── BaseAgent (direct inheritance)
    ├── BM25RetrievalAgent
    ├── RRFAgent
    ├── HierarchicalAutoMergingAgent
    ├── CrossEncoderRerankingAgent
    └── MultiHopReasoningAgent
```

## Creating a New Agent

### Step 1: Choose Base Class
```python
from radiant.agents.base_agent import BaseAgent, LLMAgent, RetrievalAgent, AgentCategory
```

### Step 2: Implement Required Properties
```python
class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MyAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY  # or PLANNING, QUERY_PROCESSING, RETRIEVAL, etc.
    
    @property
    def description(self) -> str:
        return "What this agent does"
```

### Step 3: Implement _execute()
```python
def _execute(
    self,
    query: str,           # Use descriptive parameter names
    context: List[Any],   # These names are used by orchestrator
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
    # Implementation here
    return result
```

### Step 4: Optional Lifecycle Hooks
```python
def _before_execute(self, **kwargs) -> None:
    """Called before _execute(). Use for validation/setup."""
    pass

def _after_execute(self, result: Any, metrics: AgentMetrics, **kwargs) -> Any:
    """Called after success. Can modify result."""
    metrics.custom["my_metric"] = some_value
    return result

def _on_error(self, error: Exception, metrics: AgentMetrics, **kwargs) -> Optional[Any]:
    """Called on error. Return fallback or None to propagate."""
    if isinstance(error, RecoverableError):
        return self._default_result()
    return None  # Re-raises the error
```

## Agent Parameter Reference

CRITICAL: Orchestrator calls use keyword arguments. Names must match exactly.

| Agent | _execute() Parameters |
|-------|----------------------|
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

## Adding Agent to Orchestrator

### 1. Import in orchestrator.py
```python
from radiant.agents import MyNewAgent
```

### 2. Initialize in __init__
```python
self._my_agent = MyNewAgent(llm=llm, config=config.my_agent)
```

### 3. Call with correct parameters
```python
result = self._my_agent.run(
    correlation_id=ctx.run_id,
    query=ctx.original_query,      # Must match _execute() param names
    context=some_context,
)
data = _extract_agent_data(
    result,
    default=fallback_value,
    agent_name="MyNewAgent",
    metrics_collector=self._metrics_collector,
)
```

## Testing Agents

### Unit Test Template
```python
import pytest
from radiant.agents import AgentResult, AgentStatus

class TestMyAgent:
    def test_successful_execution(self):
        agent = MyAgent(config)
        result = agent.run(query="test")
        
        assert result.success is True
        assert result.status == AgentStatus.SUCCESS
        assert result.data is not None
        assert result.metrics.duration_ms > 0
    
    def test_handles_empty_input(self):
        agent = MyAgent(config)
        result = agent.run(query="")
        
        # Verify graceful handling
        assert result.success is True or result.data is not None
```

### Run Agent Tests
```bash
pytest tests/test_base_agent_lifecycle.py -v
pytest tests/test_agents/test_my_agent.py -v
```

## Agent Categories

```python
class AgentCategory(Enum):
    PLANNING = "planning"           # Query analysis, execution planning
    QUERY_PROCESSING = "query_processing"  # Decomposition, rewrite, expansion
    RETRIEVAL = "retrieval"         # Dense, sparse, web search
    POST_RETRIEVAL = "post_retrieval"      # Fusion, reranking, merging
    GENERATION = "generation"       # Answer synthesis
    EVALUATION = "evaluation"       # Critic, fact verification
    TOOL = "tool"                   # Calculator, code execution
    UTILITY = "utility"             # General purpose
```

## Metrics Collection

### Automatic Metrics
Every agent automatically collects:
- `duration_ms` - Execution time
- `status` - SUCCESS, FAILED, PARTIAL, SKIPPED
- `run_id` - Unique execution ID
- `correlation_id` - Request tracing ID

### Custom Metrics
Add in `_after_execute()`:
```python
def _after_execute(self, result, metrics, **kwargs):
    metrics.items_processed = len(result)
    metrics.confidence = self._calculate_confidence(result)
    metrics.custom["cache_hit"] = self._cache_hit
    return result
```

### Export Metrics
```python
from radiant.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(namespace="radiant")
result = agent.run(query="test")
exporter.record_execution(result)
```

## Common Mistakes

### Wrong Parameter Name
```python
# ✗ WRONG - orchestrator uses 'retrieval_lists' but agent expects 'runs'
result = self._rrf_agent.run(retrieval_lists=data)

# ✓ CORRECT - check _execute() signature
result = self._rrf_agent.run(runs=data)
```

### Missing Correlation ID
```python
# ✗ Missing tracing
result = agent.run(query=query)

# ✓ Include for tracing
result = agent.run(correlation_id=ctx.run_id, query=query)
```

### Forgetting to Extract Data
```python
# ✗ Using AgentResult directly as data
ctx.plan = self._planning_agent.run(query=query)  # Wrong!

# ✓ Extract data from result
result = self._planning_agent.run(query=query)
ctx.plan = result.data if result.success else default_plan
```

## File Reference

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `base.py` | AgentContext and context utilities |
| `base_agent.py` | BaseAgent, LLMAgent, RetrievalAgent, AgentResult |
| `agent_template.py` | Template for creating new agents |
| `registry.py` | Agent registration and discovery |
| `planning.py` | Query analysis and execution planning |
| `decomposition.py` | Break complex queries into sub-queries |
| `rewrite.py` | Transform queries for better retrieval |
| `expansion.py` | Add synonyms and related terms |
| `dense.py` | Vector similarity retrieval |
| `bm25.py` | Sparse keyword retrieval |
| `fusion.py` | RRF fusion of multiple retrievers |
| `automerge.py` | Hierarchical chunk merging |
| `rerank.py` | Cross-encoder reranking |
| `synthesis.py` | Answer generation |
| `critic.py` | Answer quality evaluation |
| `multihop.py` | Multi-step reasoning chains |
| `web_search.py` | Web content augmentation |
| `context_eval.py` | Pre-generation context quality |
| `summarization.py` | Context compression |
| `fact_verification.py` | Claim verification |
| `citation.py` | Source attribution |
| `chunking.py` | Semantic document chunking |
| `language_detection.py` | Language identification |
| `translation.py` | Cross-language support |
| `strategy_memory.py` | Adaptive retrieval learning |
| `tools.py` | Calculator, code execution |
