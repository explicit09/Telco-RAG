"""Retrieval composition: lexical baseline, optional dense adapter, rank fusion."""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from .models import Chunk, Evidence, SearchStore


class Embedder(Protocol):
    model_id: str
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rank(self, query: str, evidence: Sequence[Evidence]) -> list[Evidence]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError("Invalid or incompatible embedding vectors")
    norm = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    return sum(x*y for x, y in zip(a, b)) / norm if norm else 0.0


def fuse_rankings(rankings: Sequence[Sequence[Evidence]], *, k: int = 60) -> list[Evidence]:
    """Reciprocal rank fusion. A channel cannot inflate a chunk through duplicates."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    channels: dict[str, set[str]] = {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            key = item.chunk.id
            if key in seen:
                continue
            seen.add(key)
            if key in chunks and chunks[key] != item.chunk:
                raise ValueError("Conflicting content for a chunk ID")
            chunks[key] = item.chunk
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            channels.setdefault(key, set()).add(item.channel)
    return [Evidence(chunks[key], score, "+".join(sorted(channels[key])))
            for key, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


class Retriever:
    def __init__(self, store: SearchStore, *, embedder: Embedder | None = None,
                 reranker: Reranker | None = None):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self._vectors: dict[tuple[str, str, str], list[float]] = {}

    def search(self, query: str, *, corpus_ids: Sequence[str], release: str | None = None,
               limit: int = 8, candidates: int = 40) -> list[Evidence]:
        if not query.strip() or not corpus_ids or limit < 1 or candidates < limit:
            raise ValueError("Search requires text, explicit corpora, and candidates >= limit > 0")
        lexical = self.store.search(query, corpus_ids=corpus_ids, release=release, limit=candidates)
        rankings = [lexical]
        if self.embedder is not None:
            # Reference exact dense scan for experiments; replace at the store seam for large corpora.
            chunks = self.store.chunks(corpus_ids, release=release)
            missing = [c for c in chunks if (self.embedder.model_id, c.id, c.text) not in self._vectors]
            if missing:
                vectors = self.embedder.embed([f"{c.title}\n{c.section}\n{c.text}" for c in missing])
                if len(vectors) != len(missing):
                    raise ValueError("Embedder returned the wrong number of vectors")
                for chunk, vector in zip(missing, vectors):
                    self._vectors[(self.embedder.model_id, chunk.id, chunk.text)] = vector
            qv = self.embedder.embed([query])
            if len(qv) != 1:
                raise ValueError("Embedder returned the wrong number of query vectors")
            dense = [Evidence(c, cosine(qv[0], self._vectors[(self.embedder.model_id, c.id, c.text)]), "dense") for c in chunks]
            rankings.append(sorted(dense, key=lambda e: (-e.score, e.chunk.id))[:candidates])
        evidence = fuse_rankings(rankings)
        if self.reranker is not None:
            reranked = self.reranker.rank(query, evidence)
            originals = {e.chunk.id: e.chunk for e in evidence}
            if any(e.chunk.id not in originals or originals[e.chunk.id] != e.chunk for e in reranked):
                raise ValueError("Reranker injected evidence outside candidate set")
            if len({e.chunk.id for e in reranked}) != len(reranked):
                raise ValueError("Reranker returned duplicate evidence")
            evidence = reranked
        return evidence[:limit]
