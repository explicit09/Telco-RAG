import json
from pathlib import Path
import tempfile
import unittest
from raglab.models import Chunk
from raglab.registry import CorpusRegistry
from raglab.retrieval import Retriever
from raglab.store import SQLiteStore


class ConjunctiveSearchTests(unittest.TestCase):
    def test_all_mode_preserves_phrase_and_scope_across_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with SQLiteStore(root / 'data.sqlite') as store:
                for id, corpus, release, text in [
                    ('hit', 'manual', '1', 'cannot provide the QoS'),
                    ('phrase-only', 'manual', '1', 'cannot provide access'),
                    ('word-only', 'manual', '1', 'QoS is available'),
                    ('foreign', 'other', '1', 'cannot provide QoS'),
                    ('release', 'manual', '2', 'cannot provide QoS')]:
                    store.replace_document(corpus, id, [Chunk(id, corpus, id, '', '', '', text, release)])
            config = root / 'registry.json'
            config.write_text(json.dumps({'manual': 'data.sqlite', 'other': 'data.sqlite'}))
            with CorpusRegistry(config) as registry:
                retrieval = Retriever(registry, phrase_search=True)
                args = dict(corpus_ids=['manual'], release='1')
                hits = retrieval.search('"cannot provide" QoS', match_mode='all', **args)
                self.assertEqual([e.chunk.id for e in hits], ['hit'])
                self.assertEqual({e.chunk.id for e in retrieval.search('"cannot provide" QoS', **args)}, {'hit', 'phrase-only', 'word-only'})
                self.assertEqual(retrieval.search('"cannot provide" absent', match_mode='all', **args), [])
                with self.assertRaises(ValueError):
                    retrieval.search('QoS', match_mode='unknown', **args)

    def test_syntax_is_quoted_not_executed_as_boolean_input(self):
        self.assertEqual(SQLiteStore._query('"A OR B" timer', 'all'), '"A OR B" AND "timer"')
        self.assertEqual(SQLiteStore._query('timer OR alarm', 'all'), '"timer" AND "OR" AND "alarm"')
        with self.assertRaises(ValueError):
            SQLiteStore._query('timer', 'unknown')

    def test_all_mode_does_not_add_unconstrained_dense_candidates(self):
        class Embedder:
            def embed(self, text):
                raise AssertionError('all-mode must not expand outside term constraints')
        with SQLiteStore(':memory:') as store:
            retrieval = Retriever(store, embedder=Embedder(), phrase_search=True)
            self.assertEqual(retrieval.search('missing term', corpus_ids=['manual'], match_mode='all'), [])
