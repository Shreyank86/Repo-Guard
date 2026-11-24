"""Operation-Level Persistent Disk Cache for RepoGuard.

Owner: Member 3 (RL Environment, Training & Model Integration)

Caches fine-grained results of individual operations:
- SEARCH: query + k + filters -> list[Evidence]
- INSPECT: file_path + start_line + end_line -> code snippet
- VERIFY: claims -> VerificationResult
- LLM_GENERATE: prompt + config -> GenerationResult
- LLM_SELF_CHECK: prompt -> probability

Backed by SQLite for cross-platform speed, thread safety, and persistence across episodes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_DB_FILE = DEFAULT_CACHE_DIR / "operations.db"


class OperationCache:
    """Thread-safe, SQLite-backed cache for individual RepoGuard operations."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._hits = 0
        self._misses = 0

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        try:
            conn.execute("PRAGMA journal_mode = DELETE")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    key_hash TEXT PRIMARY KEY,
                    op_type TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    commit_hash TEXT,
                    inputs_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_op_type ON operations(op_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_id ON operations(repo_id)")
            conn.commit()

    @staticmethod
    def compute_key(
        op_type: str,
        repo_id: str,
        commit_hash: str | None,
        inputs: dict[str, Any],
    ) -> str:
        """Deterministically hash the operation descriptor and canonical inputs."""
        canonical_input = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
        payload = f"{op_type}|{repo_id}|{commit_hash or ''}|{canonical_input}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        op_type: str,
        repo_id: str,
        commit_hash: str | None,
        inputs: dict[str, Any],
    ) -> Any | None:
        """Retrieve cached output if available; returns None on cache miss."""
        key = self.compute_key(op_type, repo_id, commit_hash, inputs)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT output_json FROM operations WHERE key_hash = ?", (key,))
            row = cur.fetchone()

        if row is not None:
            self._hits += 1
            return json.loads(row[0])

        self._misses += 1
        return None

    def set(
        self,
        op_type: str,
        repo_id: str,
        commit_hash: str | None,
        inputs: dict[str, Any],
        output: Any,
    ) -> None:
        """Store output in the persistent cache."""
        key = self.compute_key(op_type, repo_id, commit_hash, inputs)
        inputs_json = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
        output_json = json.dumps(output, sort_keys=True, ensure_ascii=False)
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO operations
                (key_hash, op_type, repo_id, commit_hash, inputs_json, output_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, op_type, repo_id, commit_hash or "", inputs_json, output_json, now),
            )
            conn.commit()

    def has(
        self,
        op_type: str,
        repo_id: str,
        commit_hash: str | None,
        inputs: dict[str, Any],
    ) -> bool:
        """Check if an operation output is present in the cache."""
        key = self.compute_key(op_type, repo_id, commit_hash, inputs)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM operations WHERE key_hash = ?", (key,))
            return cur.fetchone() is not None

    def size(self) -> int:
        """Return total count of cached operations."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM operations")
            return int(cur.fetchone()[0])

    def __len__(self) -> int:
        return self.size()

    def stats(self) -> dict[str, Any]:
        """Return cache access statistics and counts by operation type."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM operations")
            total = cur.fetchone()[0]

            cur.execute("SELECT op_type, COUNT(*) FROM operations GROUP BY op_type")
            by_type = dict(cur.fetchall())

        return {
            "total_entries": total,
            "by_type": by_type,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / (self._hits + self._misses), 4)
            if (self._hits + self._misses) > 0
            else 0.0,
        }

    def clear(self) -> None:
        """Clear all cached operations."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM operations")
            conn.commit()
        self._hits = 0
        self._misses = 0
