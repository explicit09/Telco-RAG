"""Read-only FTS ranking prototype; not used by ongoing evaluation runs.

Stream ranked matches, include all ties at the cutoff, then order equal scores by
ID to retain the existing deterministic search order. Scope before the cutoff.
"""


def ranked_ids(db, query, *, corpus_ids, release=None, limit=10):
    if not corpus_ids:
        raise ValueError('explicit corpora required')
    if limit <= 0 or not query:
        return []
    marks = ','.join('?' for _ in corpus_ids)
    sql = f'SELECT id, rank FROM chunks_fts WHERE chunks_fts MATCH ? AND corpus_id IN ({marks})'
    args = [query, *corpus_ids]
    if release is not None:
        sql += ' AND release = ?'
        args.append(release)
    sql += ' ORDER BY rank'
    rows = []
    cutoff = None
    cursor = db.execute(sql, args)
    try:
        for row in cursor:
            identifier, score = row[0], row[1]
            if cutoff is not None and score > cutoff:
                break
            rows.append((identifier, score))
            if len(rows) == limit:
                cutoff = score
    finally:
        cursor.close()
    return sorted(rows, key=lambda row: (row[1], row[0]))[:limit]
