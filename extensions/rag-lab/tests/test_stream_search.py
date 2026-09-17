import unittest
from raglab.models import Chunk
from raglab.store import SQLiteStore


class StreamingSearchTests(unittest.TestCase):
    def test_quoted_phrase_is_not_split_into_alternatives(self):
        with SQLiteStore(':memory:') as store:
            chunks=[Chunk('match','a','doc','','','','bearer control function','17',{}),
                    Chunk('distractor','a','doc','','','','bearer outside control unrelated function','17',{})]
            store.replace_document('a','doc',chunks)
            self.assertEqual([e.chunk.id for e in store.search('"bearer control function"',corpus_ids=['a'])],['match'])
            self.assertEqual(store._query('""'), '')
            self.assertEqual(store._query('timer OR "release control"'), '"timer" OR "OR" OR "release control"')

    def test_matches_exhaustive_search_with_ties_and_scopes(self):
        with SQLiteStore(':memory:') as store:
            for corpus,release in [('a','17'),('a','18'),('b','17')]:
                doc=corpus+release
                store.replace_document(corpus,doc,[Chunk(f'{doc}-{n:03}',corpus,doc,'source','title','section','timer expires' if n else 'timer',release,{}) for n in range(60,-1,-1)])
            for corpora,release in [(['a'],'17'),(['a','b'],'17'),(['b'],None)]:
                marks=','.join('?' for _ in corpora)
                sql=f'SELECT c.id, bm25(chunks_fts) AS rank FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.id WHERE chunks_fts MATCH ? AND c.corpus_id IN ({marks})'
                args=['"timer"',*corpora]
                if release:
                    sql+=' AND c.release=?';args.append(release)
                sql+=' ORDER BY rank, c.id LIMIT 40'
                expected=[(r['id'],-float(r['rank'])) for r in store.db.execute(sql,args)]
                actual=[(e.chunk.id,e.score) for e in store.search('timer',corpus_ids=corpora,release=release,limit=40)]
                self.assertEqual(actual,expected)
            self.assertEqual(store.search('absent',corpus_ids=['a']),[])


if __name__=='__main__':unittest.main()
