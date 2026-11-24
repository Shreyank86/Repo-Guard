# Retrieval Module (`src/repoguard/retrieval/`)

**Owner: Member 1** · Contract: `docs/index_schema.md` §6 · RL signals: `docs/state_signals.md` · Results: `reports/retrieval/README.md`

Stable public interface (other modules never touch FAISS/NumPy directly):

```python
from repoguard.retrieval import Retriever, SearchFilters, SignalExtractor, search

r = Retriever.from_dir("data/indexes/schedule", repo_root="data/repos/schedule")
res = r.search("cancel a scheduled job", k=5, filters=SearchFilters(exclude_tests=True))
res.evidence  # Evidence with stable evidence_id, path, lines, symbol, snippet, score, provenance
res.diagnostics  # latency, candidates, duplicates, empty, retrieved tokens
r.inspect(res.evidence[0].evidence_id)  # INSPECT/OPEN: full source of a symbol, class or file
SignalExtractor(r).compute(question)  # RL free signals -> StateVector fields
search(question, index, k=5)  # Phase 1 functional API (unchanged)
```

| File | Responsibility |
|---|---|
| `retriever.py` | `Retriever`: exact search, metadata filters, de-duplication, hybrid symbol boost, diagnostics, stable evidence ids + provenance, `get_evidence`, `inspect` |
| `signals.py` | `SignalExtractor` / `compute_free_signals`: the six StateVector free signals plus extended diagnostics |
| `query.py` | Detecting code symbols and `.py` files named in a question |
| `metrics.py` | Recall@k, Precision@k, Hit@k, MRR, nDCG@k at symbol level; `evaluate_cases` |
| `search.py` | Vector store (FAISS `IndexFlatIP` or portable NumPy), staleness checks on load, `build_vector_index`, `search()` wrapper |
| `embedder.py` | `SentenceTransformerEmbedder` (all-MiniLM-L6-v2, `pip install -e .[ml]`) and deterministic offline `HashingEmbedder` |

Scripts: `scripts/search_index.py` (search / filters / `--signals` / `--inspect` / `--freshness`), `scripts/build_retrieval_benchmark.py`, `scripts/evaluate_retrieval.py`.

The backend that built an index is recorded in `manifest.json` (`tool_versions`) and `vector_meta.json`; queries always use the same backend. Vectors that don't match the index's passages raise `StaleIndexError` instead of returning wrong evidence.
