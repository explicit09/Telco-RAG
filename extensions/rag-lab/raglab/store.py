"""SQLite FTS5 search store."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Sequence

from .models import Chunk, Evidence


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL, document_id TEXT NOT NULL,
                source TEXT NOT NULL, title TEXT NOT NULL, section TEXT NOT NULL,
                text TEXT NOT NULL, release TEXT NOT NULL, metadata TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_corpus_document ON chunks(corpus_id, document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED, corpus_id UNINDEXED, document_id UNINDEXED,
                source, title, section, text, release UNINDEXED
            );
        """)
        self.db.commit()

    def replace_document(self, corpus_id: str, document_id: str, chunks: Sequence[Chunk]) -> None:
        entries = list(chunks)
        ids: set[str] = set()
        for chunk in entries:
            if chunk.corpus_id != corpus_id or chunk.document_id != document_id:
                raise ValueError("all chunks must belong to the replaced corpus and document")
            if not chunk.id or chunk.id in ids:
                raise ValueError("chunk ids must be non-empty and unique")
            ids.add(chunk.id)
            try:
                json.dumps(chunk.metadata, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("chunk metadata must be JSON serializable") from exc
        with self.db:
            for chunk in entries:
                owner = self.db.execute("SELECT corpus_id, document_id FROM chunks WHERE id = ?", (chunk.id,)).fetchone()
                if owner and (owner["corpus_id"], owner["document_id"]) != (corpus_id, document_id):
                    raise ValueError(f"chunk id already belongs to another document: {chunk.id}")
            exists = self.db.execute("SELECT 1 FROM chunks WHERE corpus_id = ? AND document_id = ? LIMIT 1", (corpus_id, document_id)).fetchone()
            if exists:
                self.db.execute("DELETE FROM chunks_fts WHERE corpus_id = ? AND document_id = ?", (corpus_id, document_id))
            self.db.execute("DELETE FROM chunks WHERE corpus_id = ? AND document_id = ?", (corpus_id, document_id))
            for chunk in entries:
                self.db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (chunk.id, chunk.corpus_id, chunk.document_id, chunk.source, chunk.title, chunk.section, chunk.text, chunk.release, json.dumps(chunk.metadata, sort_keys=True, separators=(",", ":"))))
                self.db.execute("INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (chunk.id, chunk.corpus_id, chunk.document_id, chunk.source, chunk.title, chunk.section, chunk.text, chunk.release))

    @staticmethod
    def _query(query: str, match_mode: str = "any") -> str:
        if match_mode not in ("any", "all"):
            raise ValueError("match_mode must be any or all")
        terms = []
        for phrase, word in re.findall(r'"([^\"]*)"|([\w]+)', query, re.UNICODE):
            tokens = re.findall(r"[\w]+", phrase or word, re.UNICODE)
            if tokens:
                terms.append('"' + " ".join(tokens) + '"')
        return (" AND " if match_mode == "all" else " OR ").join(terms)

    def search(self, query: str, *, corpus_ids: Sequence[str], release: str | None = None, limit: int = 10, match_mode: str = "any") -> list[Evidence]:
        corpora = list(corpus_ids)
        if not corpora:
            raise ValueError("corpus_ids must be non-empty")
        compiled_query = self._query(query, match_mode)
        if limit <= 0 or not compiled_query:
            return []
        marks = ",".join("?" for _ in corpora)
        args: list[object] = [compiled_query, *corpora]
        sql = f"SELECT id, rank FROM chunks_fts WHERE chunks_fts MATCH ? AND corpus_id IN ({marks})"
        if release is not None:
            sql += " AND release = ?"
            args.append(release)
        sql += " ORDER BY rank"
        # FTS5 streams by rank. Include the whole cutoff tie, then sort IDs to
        # preserve the exhaustive query's deterministic order without a full join.
        ranked = []
        cutoff = None
        cursor = self.db.execute(sql, args)
        try:
            for row in cursor:
                if cutoff is not None and row["rank"] > cutoff:
                    break
                ranked.append((row["id"], float(row["rank"])))
                if len(ranked) == limit:
                    cutoff = row["rank"]
        finally:
            cursor.close()
        result = []
        for identifier, rank in sorted(ranked, key=lambda item: (item[1], item[0]))[:limit]:
            chunk = self.get_chunk(identifier)
            if chunk is None or chunk.corpus_id not in corpora or (release is not None and chunk.release != release):
                raise ValueError("search index and chunk metadata disagree")
            result.append(Evidence(chunk, score=-rank))
        return result

    def read_window(self, anchor_id: str, *, corpus_ids: Sequence[str], release: str | None = None,
                    before: int = 3, after: int = 3, limit: int = 16, max_chars: int = 24000) -> list[Evidence]:
        """Read bounded neighboring paragraphs in the anchor's document and section.

        Split pieces with the same ordinal use deterministic ID order. This is
        a passage window, not a reconstruction of the original document layout.
        """
        if not corpus_ids or type(before) is not int or type(after) is not int or not 0 <= before <= 8 or not 0 <= after <= 8:
            raise ValueError("read window requires corpora and 0..8 neighboring ordinals")
        if type(limit) is not int or not 1 <= limit <= 16 or type(max_chars) is not int or not 1 <= max_chars <= 24000:
            raise ValueError("read window exceeds passage or character budget")
        anchor = self.get_chunk(anchor_id)
        if anchor is None or anchor.corpus_id not in corpus_ids or (release is not None and anchor.release != release):
            raise ValueError("read anchor is outside the requested scope")
        ordinal = anchor.metadata.get("ordinal")
        if type(ordinal) is not int or ordinal < 1:
            raise ValueError("read anchor has no valid paragraph ordinal")
        rows = self.db.execute("""SELECT id FROM chunks
            WHERE corpus_id=? AND document_id=? AND release=? AND section=?
            AND json_type(metadata, '$.ordinal')='integer'
            AND json_extract(metadata, '$.ordinal') BETWEEN ? AND ?
            ORDER BY (id=?) DESC, ABS(json_extract(metadata, '$.ordinal')-?), json_extract(metadata, '$.ordinal'), id
            LIMIT ?""", (anchor.corpus_id, anchor.document_id, anchor.release, anchor.section,
                           max(1, ordinal-before), ordinal+after, anchor_id, ordinal, limit))
        result = []
        used = 0
        for row in rows:
            chunk = self.get_chunk(row["id"])
            if used + len(chunk.text) > max_chars:
                continue
            result.append(Evidence(chunk, 0.0, "document_read"))
            used += len(chunk.text)
        return sorted(result, key=lambda e: (e.chunk.metadata["ordinal"], e.chunk.id))

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        r = self.db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return None if r is None else Chunk(r["id"], r["corpus_id"], r["document_id"], r["source"], r["title"], r["section"], r["text"], r["release"], json.loads(r["metadata"]))

    def list_corpora(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT corpus_id, COUNT(*) AS count FROM chunks GROUP BY corpus_id ORDER BY corpus_id")]

    def chunks(self, corpus_ids: Sequence[str], release: str | None = None) -> list[Chunk]:
        if not corpus_ids:
            raise ValueError("corpus_ids must be non-empty")
        marks = ",".join("?" for _ in corpus_ids)
        args: list[object] = list(corpus_ids)
        sql = f"SELECT * FROM chunks WHERE corpus_id IN ({marks})"
        if release is not None:
            sql += " AND release = ?"
            args.append(release)
        sql += " ORDER BY corpus_id, document_id, id"
        return [self.get_chunk(r["id"]) for r in self.db.execute(sql, args).fetchall()]

    def fingerprint(self, corpus_ids: Sequence[str]) -> str:
        rows = self.chunks(corpus_ids)
        canonical = [{"id": c.id, "corpus_id": c.corpus_id, "document_id": c.document_id, "source": c.source, "title": c.title, "section": c.section, "text": c.text, "release": c.release, "metadata": c.metadata} for c in rows]
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
