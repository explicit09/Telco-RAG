"""Explicit corpus-to-database routing, with no fallback to another corpus."""
import json
from pathlib import Path

from .retrieval import fuse_rankings
from .store import SQLiteStore


class CorpusRegistry:
    def __init__(self, path):
        self.path = Path(path).resolve()
        config = json.loads(self.path.read_text())
        if not isinstance(config, dict) or not config:
            raise ValueError('registry must map corpus IDs to database paths')
        self.paths = {}
        for corpus, filename in config.items():
            if not isinstance(corpus, str) or not corpus or not isinstance(filename, str):
                raise ValueError('invalid corpus registry entry')
            database = (self.path.parent / filename).resolve()
            if not database.is_file():
                raise FileNotFoundError(f'missing database for {corpus}: {database}')
            self.paths[corpus] = database
        self.stores = {}

    def _store(self, corpus):
        if corpus not in self.paths:
            raise ValueError(f'corpus is not registered: {corpus}')
        if corpus not in self.stores:
            self.stores[corpus] = SQLiteStore(self.paths[corpus])
        return self.stores[corpus]

    def search(self, query, *, corpus_ids, release=None, limit=10):
        if not corpus_ids:
            raise ValueError('corpus_ids must be non-empty')
        rankings = [self._store(c).search(query, corpus_ids=[c], release=release, limit=limit) for c in dict.fromkeys(corpus_ids)]
        return fuse_rankings(rankings)[:limit]

    def get_chunk(self, identifier):
        # Search only databases opened by explicit corpus requests.
        found = []
        for corpus, store in self.stores.items():
            chunk = store.get_chunk(identifier)
            if chunk is not None:
                if chunk.corpus_id != corpus:
                    raise ValueError('chunk belongs to a different corpus')
                found.append(chunk)
        if len(found) > 1:
            raise ValueError('ambiguous chunk ID across databases')
        return found[0] if found else None

    def provenance(self, corpus_ids):
        result = {}
        for corpus in sorted(set(corpus_ids)):
            if corpus not in self.paths:
                raise ValueError(f'corpus is not registered: {corpus}')
            manifest = self.paths[corpus].with_suffix('.manifest.json')
            record = json.loads(manifest.read_text())
            if record.get('corpus') != corpus:
                raise ValueError('database manifest corpus mismatch')
            result[corpus] = record
        return result

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
