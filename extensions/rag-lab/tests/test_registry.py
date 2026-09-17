import json
from pathlib import Path
import tempfile
import unittest
from raglab.ingest import ingest_file
from raglab.store import SQLiteStore
from raglab.registry import CorpusRegistry


class RegistryTests(unittest.TestCase):
    def test_identical_document_names_in_separate_databases_stay_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config={}
            for corpus, release, text in [('radio','17','timer is 5 seconds'),('manual','1','timer is 30 seconds')]:
                source=root/'same.txt';source.write_text(text)
                with SQLiteStore(root/f'{corpus}.sqlite') as store:
                    store.replace_document(corpus,'same',ingest_file(source,corpus_id=corpus,document_id='same',release=release))
                config[corpus]=f'{corpus}.sqlite'
            path=root/'registry.json';path.write_text(json.dumps(config))
            with CorpusRegistry(path) as registry:
                results=registry.search('timer',corpus_ids=['radio'],release='17')
                self.assertEqual(len(results),1)
                self.assertIn('5 seconds',results[0].chunk.text)
                self.assertEqual(registry.get_chunk(results[0].chunk.id),results[0].chunk)
                self.assertEqual(registry.search('timer',corpus_ids=['radio'],release='18'),[])
                with self.assertRaises(ValueError):
                    registry.search('timer',corpus_ids=['missing'])
                self.assertEqual(len(registry.search('timer',corpus_ids=['radio','manual'])),2)

    def test_supplements_are_searched_without_mutating_baseline(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,text in [('base','timer expires'),('supplement','timer restarts')]:
                source=root/f'{name}.txt';source.write_text(text)
                with SQLiteStore(root/f'{name}.sqlite') as store:
                    store.replace_document('radio',name,ingest_file(source,corpus_id='radio',document_id=name,release='17'))
                (root/f'{name}.manifest.json').write_text(json.dumps({'corpus':'radio','release':'17','source':name}))
            baseline_hash=hashlib.sha256((root/'base.sqlite').read_bytes()).hexdigest()
            config=root/'registry.json';config.write_text(json.dumps({'radio':['base.sqlite','supplement.sqlite']}))
            with CorpusRegistry(config) as registry:
                results=registry.search('timer',corpus_ids=['radio'],release='17')
                self.assertEqual({e.chunk.document_id for e in results},{'base','supplement'})
                self.assertTrue(all(registry.get_chunk(e.chunk.id)==e.chunk for e in results))
                self.assertEqual(registry.search('timer',corpus_ids=['radio'],release='18'),[])
                self.assertEqual(len(registry.provenance(['radio'])['radio']['parts']),2)
            self.assertEqual(hashlib.sha256((root/'base.sqlite').read_bytes()).hexdigest(),baseline_hash)
            config.write_text(json.dumps({'radio':['base.sqlite','base.sqlite']}))
            with self.assertRaisesRegex(ValueError,'duplicate database'):
                CorpusRegistry(config)

    def test_shared_database_exposes_only_requested_corpora(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);chunks={}
            with SQLiteStore(root/'shared.sqlite') as store:
                for corpus in ('a','b'):
                    source=root/f'{corpus}.txt';source.write_text('shared timer')
                    chunks[corpus]=ingest_file(source,corpus_id=corpus,release='1')[0]
                    store.replace_document(corpus,chunks[corpus].document_id,[chunks[corpus]])
            config=root/'registry.json';config.write_text(json.dumps({'a':'shared.sqlite','b':'shared.sqlite'}))
            with CorpusRegistry(config) as registry:
                registry.search('timer',corpus_ids=['a'])
                self.assertIsNone(registry.get_chunk(chunks['b'].id))
                registry.search('timer',corpus_ids=['b'])
                self.assertEqual(registry.get_chunk(chunks['b'].id),chunks['b'])
