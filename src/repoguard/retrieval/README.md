# Retrieval Module (`src/repoguard/retrieval/`)

**Owner: Member 1** · see `src/repoguard/repository/README.md` and `docs/index_schema.md` §6.

- `embedder.py` — `SentenceTransformerEmbedder` (`sentence-transformers/all-MiniLM-L6-v2`, `pip install -e .[ml]`) and `HashingEmbedder` (deterministic, offline, lexical). `get_embedder({"backend": "auto"|"sentence-transformers"|"hashing"})`.
- `search.py` — `build_vector_index()` (FAISS `IndexFlatIP` over normalised vectors, NumPy fallback) and `search(question, index, k) -> list[Evidence]`.

The backend that built an index is recorded in `manifest.json` and `vector_meta.json`; queries are always embedded with the same backend.
