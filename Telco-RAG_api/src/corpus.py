"""Corpus boundary for Telco-RAG's two-pass retrieval flow.

Providers own storage and ranking. Query owns candidate generation and validation.
No answer keys or evaluation records are accepted by this interface.
"""
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str


class Corpus(Protocol):
    def search(self, query: str, *, limit: int) -> Sequence[Passage]: ...


class StoreCorpus:
    """Adapt a scoped search store, including the local raglab SQLite store."""
    def __init__(self, store, *, corpus_ids, release=None):
        if not corpus_ids:
            raise ValueError("at least one corpus must be selected")
        self.store = store
        self.corpus_ids = tuple(corpus_ids)
        self.release = release

    def search(self, query: str, *, limit: int):
        return [Passage(item.chunk.id, item.chunk.text,
                        f"{item.chunk.corpus_id}/{item.chunk.source} [{item.chunk.section}]")
                for item in self.store.search(query, corpus_ids=self.corpus_ids,
                                              release=self.release, limit=limit)]
