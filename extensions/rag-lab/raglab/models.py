from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Chunk:
    id: str
    corpus_id: str
    document_id: str
    source: str
    title: str
    section: str
    text: str
    release: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    chunk: Chunk
    score: float
    channel: str = "lexical"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Question:
    """Only public inputs. Gold answers must never enter this object."""
    id: str
    text: str
    options: dict[str, str] = field(default_factory=dict)
    corpus_ids: tuple[str, ...] = ()
    release: str | None = None


@dataclass(frozen=True)
class Answer:
    question_id: str
    text: str
    selected_option: str | None
    citations: tuple[str, ...]
    abstained: bool
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class SearchStore(Protocol):
    def replace_document(self, corpus_id: str, document_id: str, chunks: Sequence[Chunk]) -> None: ...
    def search(self, query: str, *, corpus_ids: Sequence[str], release: str | None = None, limit: int = 10) -> list[Evidence]: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def list_corpora(self) -> list[dict]: ...
    def chunks(self, corpus_ids: Sequence[str], release: str | None = None) -> list[Chunk]: ...
    def fingerprint(self, corpus_ids: Sequence[str]) -> str: ...
    def close(self) -> None: ...


class Generator(Protocol):
    def generate(self, question: Question, evidence: Sequence[Evidence]) -> Answer: ...
