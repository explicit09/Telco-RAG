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
        for corpus, filenames in config.items():
            if isinstance(filenames, str):
                filenames = [filenames]
            if (not isinstance(corpus, str) or not corpus or not isinstance(filenames, list)
                    or not filenames or any(not isinstance(name, str) or not name for name in filenames)):
                raise ValueError('invalid corpus registry entry')
            databases = tuple((self.path.parent / name).resolve() for name in filenames)
            if len(set(databases)) != len(databases):
                raise ValueError('duplicate database for corpus')
            for database in databases:
                if not database.is_file():
                    raise FileNotFoundError(f'missing database for {corpus}: {database}')
            self.paths[corpus] = databases
        self.stores = {}
        self.active_corpora = {}

    def _stores_for(self, corpus):
        if corpus not in self.paths:
            raise ValueError(f'corpus is not registered: {corpus}')
        for database in self.paths[corpus]:
            if database not in self.stores:
                self.stores[database] = SQLiteStore(database)
            self.active_corpora.setdefault(database, set()).add(corpus)
            yield self.stores[database]

    def search(self, query, *, corpus_ids, release=None, limit=10):
        if not corpus_ids:
            raise ValueError('corpus_ids must be non-empty')
        if limit <= 0:
            return []
        rankings = [store.search(query, corpus_ids=[corpus], release=release, limit=limit)
                    for corpus in dict.fromkeys(corpus_ids) for store in self._stores_for(corpus)]
        return fuse_rankings(rankings)[:limit]

    def read_window(self, anchor_id, *, corpus_ids, release=None, before=3, after=3,
                    limit=16, max_chars=24000):
        if not corpus_ids:
            raise ValueError('corpus_ids must be non-empty')
        owners, seen = [], set()
        for corpus in dict.fromkeys(corpus_ids):
            for store in self._stores_for(corpus):
                key = (id(store), corpus)
                if key in seen:
                    continue
                seen.add(key)
                chunk = store.get_chunk(anchor_id)
                if chunk is not None and chunk.corpus_id == corpus and (release is None or chunk.release == release):
                    owners.append((store, chunk))
        if len(owners) != 1:
            raise ValueError('read anchor must have exactly one owner in the requested scope')
        store, chunk = owners[0]
        return store.read_window(anchor_id, corpus_ids=[chunk.corpus_id], release=release,
                                 before=before, after=after, limit=limit, max_chars=max_chars)

    def get_chunk(self, identifier):
        # Only previously requested corpora in opened databases are eligible.
        found = []
        for database, store in self.stores.items():
            chunk = store.get_chunk(identifier)
            if chunk is not None and chunk.corpus_id in self.active_corpora[database]:
                found.append(chunk)
        if len(found) > 1:
            raise ValueError('ambiguous chunk ID across databases')
        return found[0] if found else None

    def provenance(self, corpus_ids):
        result = {}
        for corpus in sorted(set(corpus_ids)):
            if corpus not in self.paths:
                raise ValueError(f'corpus is not registered: {corpus}')
            records = []
            for database in self.paths[corpus]:
                record = json.loads(database.with_suffix('.manifest.json').read_text())
                if record.get('corpus') != corpus:
                    raise ValueError('database manifest corpus mismatch')
                records.append(record)
            result[corpus] = records[0] if len(records) == 1 else {'corpus': corpus, 'parts': records}
        return result

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
        self.active_corpora.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
