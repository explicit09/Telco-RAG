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
    def _query(query: str) -> str:
        return " OR ".join('"' + token.replace('"', '""') + '"' for token in re.findall(r"[\w]+", query, re.UNICODE))

    def search(self, query: str, *, corpus_ids: Sequence[str], release: str | None = None, limit: int = 10) -> list[Evidence]:
        corpora = list(corpus_ids)
        if not corpora:
            raise ValueError("corpus_ids must be non-empty")
        if limit <= 0 or not self._query(query):
            return []
        marks = ",".join("?" for _ in corpora)
        args: list[object] = [self._query(query), *corpora]
        release_sql = ""
        if release is not None:
            release_sql = " AND c.release = ?"
            args.append(release)
        args.append(limit)
        rows = self.db.execute(f"SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.id WHERE chunks_fts MATCH ? AND c.corpus_id IN ({marks}){release_sql} ORDER BY rank, c.id LIMIT ?", args).fetchall()
        return [Evidence(Chunk(r["id"], r["corpus_id"], r["document_id"], r["source"], r["title"], r["section"], r["text"], r["release"], json.loads(r["metadata"])), score=-float(r["rank"])) for r in rows]

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
