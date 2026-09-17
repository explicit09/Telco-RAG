"""Index a verified source manifest with document-level resume checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'extensions/rag-lab'))
from raglab.ingest import ingest_file
from raglab.store import SQLiteStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('database', type=Path)
    p.add_argument('--corpus', required=True)
    p.add_argument('--release', default='')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.database.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with SQLiteStore(a.database) as store:
        store.db.execute('CREATE TABLE IF NOT EXISTS ingested_sources(corpus TEXT, document TEXT, sha256 TEXT, parser_sha256 TEXT, release TEXT, PRIMARY KEY(corpus, document))')
        parser_source = (ROOT / 'extensions/rag-lab/raglab/ingest.py').read_bytes()
        parser_hash = hashlib.sha256(parser_source).hexdigest()
        snapshots = a.database.parent / 'parser-snapshots'
        snapshots.mkdir(exist_ok=True)
        (snapshots / (parser_hash + '.py')).write_bytes(parser_source)
        for index, entry in enumerate(manifest['files'], 1):
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe manifest path')
            source = a.manifest.parent / relative
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != entry['sha256']:
                raise ValueError(f'source hash mismatch: {relative}')
            old = store.db.execute('SELECT sha256,parser_sha256,release FROM ingested_sources WHERE corpus=? AND document=?', (a.corpus, str(relative))).fetchone()
            if old is None or tuple(old) != (digest, parser_hash, a.release):
                chunks = ingest_file(source, corpus_id=a.corpus, document_id=str(relative), release=a.release)
                store.replace_document(a.corpus, str(relative), chunks)
                with store.db:
                    store.db.execute('INSERT OR REPLACE INTO ingested_sources VALUES(?,?,?,?,?)', (a.corpus, str(relative), digest, parser_hash, a.release))
            if index % 25 == 0 or index == len(manifest['files']):
                print(f'indexed {index}/{len(manifest["files"])} documents', flush=True)
        count = store.db.execute('SELECT count(*) FROM chunks WHERE corpus_id=?', (a.corpus,)).fetchone()[0]
    summary = {'corpus': a.corpus, 'release': a.release, 'documents': len(manifest['files']), 'chunks': count, 'parser_sha256': parser_hash, 'source_manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(), 'seconds': round(time.monotonic()-start, 2)}
    a.database.with_suffix('.manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
