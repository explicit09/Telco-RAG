"""Retrieval composition: lexical baseline, optional dense adapter, rank fusion."""
from __future__ import annotations

import math
from itertools import zip_longest
import re
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
                 reranker: Reranker | None = None, phrase_search: bool = False,
                 rerank_strategy: str = "replace"):
        if rerank_strategy not in ("replace", "interleave"):
            raise ValueError("Unknown rerank strategy")
        self.rerank_strategy = rerank_strategy
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.phrase_search = phrase_search
        self._vectors: dict[tuple[str, str, str], list[float]] = {}

    def read_window(self, anchor_id: str, *, corpus_ids: Sequence[str], release: str | None = None,
                    before: int = 3, after: int = 3, limit: int = 16, max_chars: int = 24000) -> list[Evidence]:
        # A source window keeps source order and is not reranked as search results.
        return self.store.read_window(anchor_id, corpus_ids=corpus_ids, release=release,
                                      before=before, after=after, limit=limit, max_chars=max_chars)

    def search(self, query: str, *, corpus_ids: Sequence[str], release: str | None = None,
               limit: int = 8, candidates: int = 40, match_mode: str = "any") -> list[Evidence]:
        if match_mode not in ("any", "all"):
            raise ValueError("match_mode must be any or all")
        if not query.strip() or not corpus_ids or limit < 1 or candidates < limit:
            raise ValueError("Search requires text, explicit corpora, and candidates >= limit > 0")
        search_options = {"match_mode": "all"} if match_mode == "all" else {}
        lexical = self.store.search(query, corpus_ids=corpus_ids, release=release, limit=candidates, **search_options)
        rankings = [lexical]
        if self.phrase_search and match_mode == "any":
            phrases = {}
            for raw in re.findall(r'"([^\"]+)"', query):
                words = re.findall(r'\w+', raw)
                phrase = ' '.join(words)
                if len(words) >= 2 and len(phrase) <= 200:
                    phrases.setdefault(phrase.casefold(), phrase)
            # Reserve candidate coverage for named phrases instead of letting
            # common surrounding words crowd them out of the lexical pool.
            for phrase in list(phrases.values())[:2]:
                focused = '"' + phrase + '"'
                if focused.casefold() == query.strip().casefold():
                    continue
                matches = self.store.search(focused, corpus_ids=corpus_ids, release=release, limit=candidates)
                rankings.append([Evidence(item.chunk, item.score, 'phrase') for item in matches])
        if self.embedder is not None and match_mode == "any":
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
            if self.rerank_strategy == "interleave":
                # Preserve both candidate and reranker order without comparing
                # incompatible scores or increasing the final passage budget.
                combined = {}
                for pair in zip_longest(evidence, reranked):
                    for item in pair:
                        if item is not None:
                            combined.setdefault(item.chunk.id, item)
                evidence = list(combined.values())
            else:
                evidence = reranked
        return evidence[:limit]
