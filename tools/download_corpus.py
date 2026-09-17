"""Download a pinned public corpus, verify upstream hashes, and write a manifest."""
import argparse
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

REVISION = "b8d598e50cada8aaa4de641abbec77bef6b51839"
BASE = f"https://huggingface.co/datasets/netop/3GPP-R18/resolve/{REVISION}/"


def retry_delay(error, attempt):
    """Honor server cooldowns; otherwise back off on throttling and outages."""
    fallback = min(60, 2 ** attempt)
    if isinstance(error, urllib.error.HTTPError):
        if error.code not in (429, 500, 502, 503, 504):
            raise error
        if error.code == 429:
            fallback = max(60, fallback)
        value = error.headers.get('Retry-After') if error.headers else None
        if value:
            try:
                return max(fallback, float(value))
            except ValueError:
                try:
                    return max(fallback, parsedate_to_datetime(value).timestamp() - time.time())
                except (ValueError, TypeError, OverflowError):
                    pass
    return fallback


def verified(path, row):
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    data = path.read_bytes()
    if 'lfs' in row:
        return hashlib.sha256(data).hexdigest() == row['lfs']['oid']
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == row['oid']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inventory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--attempts', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1:
        parser.error('workers and attempts must be positive')
    inventory = json.loads(args.inventory.read_text())
    mirrored = isinstance(inventory, dict)
    repository = inventory['repository'] if mirrored else 'netop/3GPP-R18'
    revision = inventory['revision'] if mirrored else REVISION
    source_rows = inventory['files'] if mirrored else inventory
    rows = sorted([r for r in source_rows if r['path'].endswith(('.docx', '/raw.md'))], key=lambda r: r['path'])
    base = f'https://huggingface.co/datasets/{repository}/resolve/{revision}/'
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        relative = Path(row['path']) if mirrored else Path(row['path']).name
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe source path')
        name = str(relative)
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not verified(path, row):
            for attempt in range(args.attempts):
                try:
                    with urllib.request.urlopen(base + urllib.parse.quote(row['path']), timeout=90) as response:
                        data = response.read()
                    temp = path.with_suffix('.partial')
                    temp.write_bytes(data)
                    if not verified(temp, row):
                        raise ValueError(f"upstream hash mismatch: {name}")
                    temp.replace(path)
                    break
                except Exception as error:
                    if attempt == args.attempts - 1:
                        raise
                    delay = retry_delay(error, attempt)
                    print(f'retrying {name} after {delay:.0f}s ({type(error).__name__})', flush=True)
                    time.sleep(delay)
        return {'path': name, 'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'upstream_path': row['path']}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, row): row for row in rows}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0 or len(results) == len(rows):
                print(f"verified {len(results)}/{len(rows)} documents", flush=True)
    manifest = {'repository': repository, 'revision': revision, 'files': sorted(results, key=lambda r: r['path'])}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
