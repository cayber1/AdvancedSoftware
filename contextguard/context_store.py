"""
ContextGuard — Context Store
Provides versioned, traceable context management (proposal §MCP Governance).
Every context snapshot gets a version ID so the MCP log can reference it.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextVersion:
    version_id: str
    created_at: float
    documents: list[str]
    source_hash: str        # SHA-256 of joined docs — detects tampering
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "num_docs": len(self.documents),
            "source_hash": self.source_hash,
            "metadata": self.metadata,
        }


class ContextStore:
    """
    Immutable, append-only context registry.
    - commit()   → store a new context snapshot, return its version_id
    - checkout() → retrieve a snapshot by version_id
    - diff()     → detect changes between two versions (tamper detection)
    """

    def __init__(self):
        self._store: dict[str, ContextVersion] = {}
        self._history: list[str] = []          # ordered list of version_ids

    def _hash_docs(self, documents: list[str]) -> str:
        raw = "||".join(sorted(documents))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def commit(self, documents: list[str], metadata: Optional[dict] = None) -> str:
        """Store a new snapshot. Returns version_id."""
        version_id = str(uuid.uuid4())[:8]
        snapshot = ContextVersion(
            version_id=version_id,
            created_at=time.time(),
            documents=list(documents),
            source_hash=self._hash_docs(documents),
            metadata=metadata or {},
        )
        self._store[version_id] = snapshot
        self._history.append(version_id)
        return version_id

    def checkout(self, version_id: str) -> Optional[ContextVersion]:
        return self._store.get(version_id)

    def latest(self) -> Optional[ContextVersion]:
        if not self._history:
            return None
        return self._store[self._history[-1]]

    def diff(self, vid_a: str, vid_b: str) -> dict:
        """Returns which documents were added/removed between two versions."""
        a = self._store.get(vid_a)
        b = self._store.get(vid_b)
        if not a or not b:
            return {"error": "unknown version_id"}
        set_a, set_b = set(a.documents), set(b.documents)
        return {
            "added":   list(set_b - set_a),
            "removed": list(set_a - set_b),
            "unchanged": len(set_a & set_b),
            "tampered": a.source_hash != self._hash_docs(a.documents),
        }

    def history(self) -> list[dict]:
        return [self._store[vid].as_dict() for vid in self._history]
