from pathlib import Path
import sqlite3
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prototype_rank_stream import ranked_ids


class RankStreamTests(unittest.TestCase):
    def test_scope_and_all_cutoff_ties_match_exhaustive_order(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, corpus_id UNINDEXED, document_id UNINDEXED, source, title, section, text, release UNINDEXED)')
            # Reverse ID insertion order and put >limit tied matches at the cutoff.
            for n in range(60, -1, -1):
                for corpus, release in [('a','17'), ('a','18'), ('b','17')]:
                    db.execute('INSERT INTO chunks_fts VALUES (?,?,?,?,?,?,?,?)', (f'{corpus}-{release}-{n:03}',corpus,'doc','','','','timer expires' if n else 'timer',release))
            for corpora, release in [(['a'],'17'), (['a','b'],'17'), (['b'],None)]:
                marks = ','.join('?' for _ in corpora)
                sql = f'SELECT id, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? AND corpus_id IN ({marks})'
                args = ['"timer"', *corpora]
                if release:
                    sql += ' AND release = ?'; args.append(release)
                sql += ' ORDER BY score, id LIMIT 40'
                expected = db.execute(sql,args).fetchall()
                self.assertEqual(ranked_ids(db,'"timer"',corpus_ids=corpora,release=release,limit=40), expected)
            self.assertEqual(ranked_ids(db,'"absent"',corpus_ids=['a']), [])
            self.assertEqual(ranked_ids(db,'"timer"',corpus_ids=['a'],limit=0), [])


if __name__ == '__main__':
    unittest.main()
