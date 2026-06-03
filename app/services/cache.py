"""
services/cache.py
=================
SEMANTIC CACHE (a performance/cost optimisation that sits IN FRONT of retrieval).

Plain caches key on exact strings, so "What is RAG?" and "what's RAG" miss each
other. A SEMANTIC cache keys on the query EMBEDDING: if a new question is within
CACHE_SIMILARITY_THRESHOLD cosine of a cached one (in the same scope), we replay
the stored answer + sources instead of re-running retrieval and the LLM.

WHY SCOPED: a cached answer is only valid for the same set of documents. We key
each cache entry by a "scope" string derived from the requested doc_ids, so a
question scoped to doc A never returns an answer computed over docs A+B+C.

WHEN TO TUNE / REPLACE:
  • Threshold too low → wrong answers served (false hits). Too high → few hits.
    0.95 is conservative; measure hit-rate vs. user complaints and adjust.
  • This brute-forces cosine over all keys in a scope (O(n) per lookup). Fine for
    hundreds of entries; for scale, store cache embeddings in a vector index too
    (e.g. a dedicated Pinecone namespace or Redis Vector Search) and ANN-search them.
  • Consider GPTCache (a purpose-built semantic cache library) if this grows.
"""

import json
import uuid
import logging
from typing import Optional, List

import numpy as np

from app.core.config import (
    redis_client,
    CACHE_PREFIX,
    CACHE_INDEX_PREFIX,
    CACHE_TTL_SECONDS,
    CACHE_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def scope_key(doc_ids: List[str]) -> str:
    """Stable identifier for the set of sources being searched (order-independent)."""
    return "|".join(sorted(doc_ids)) if doc_ids else "__all__"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


async def lookup(query_embedding: List[float], scope: str) -> Optional[dict]:
    """Return the cached payload of a semantically-similar query, or None."""
    try:
        cache_keys = await redis_client.smembers(f"{CACHE_INDEX_PREFIX}{scope}")
        if not cache_keys:
            return None

        best_sim, best_payload = 0.0, None
        for key in cache_keys:
            raw = await redis_client.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            sim = _cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best_payload = sim, entry

        return best_payload if best_sim >= CACHE_SIMILARITY_THRESHOLD else None
    except Exception:
        # Cache must NEVER break a request — degrade gracefully to a cache miss.
        logger.warning("Redis cache lookup failed — proceeding without cache")
        return None


async def save(query_embedding: List[float], scope: str, sources: list, answer: str) -> None:
    """Persist a Q&A pair to the semantic cache under the given scope."""
    try:
        cache_key = f"{CACHE_PREFIX}{uuid.uuid4()}"
        index_key = f"{CACHE_INDEX_PREFIX}{scope}"
        payload = json.dumps(
            {"embedding": query_embedding, "sources": sources, "answer": answer}
        )
        await redis_client.setex(cache_key, CACHE_TTL_SECONDS, payload)
        await redis_client.sadd(index_key, cache_key)
        await redis_client.expire(index_key, CACHE_TTL_SECONDS)
    except Exception:
        logger.warning("Redis cache save failed — result not cached")
