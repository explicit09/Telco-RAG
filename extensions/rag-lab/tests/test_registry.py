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
