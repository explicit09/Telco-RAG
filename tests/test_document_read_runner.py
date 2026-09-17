from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_development as runner
from raglab.models import Answer, Chunk
from raglab.store import SQLiteStore


class DocumentReadRunnerTests(unittest.TestCase):
    def test_opt_in_reaches_model_and_followup_and_is_frozen_in_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            questions = root / 'dev.questions.jsonl'
            questions.write_text(json.dumps({'id': 'q', 'text': 'keyword', 'options': {'A': 'five'},
                                            'corpus_ids': ['manual'], 'release': '1'}) + '\n')
            database = root / 'manual.sqlite'
            with SQLiteStore(database) as store:
                store.replace_document('manual', 'doc', [
                    Chunk('anchor', 'manual', 'doc', 'source', 'Manual', 'section', 'keyword: see next paragraph.', '1', {'ordinal': 1}),
                    Chunk('value', 'manual', 'doc', 'source', 'Manual', 'section', 'The value is five.', '1', {'ordinal': 2}),
                ])
            database.with_suffix('.manifest.json').write_text(json.dumps({'corpus': 'manual', 'release': '1'}))
            output = root / 'run.json'
            configured = []

            class FakeGenerator:
                requests = 0

                def __init__(self, **kwargs):
                    configured.append(kwargs)

                def generate(self, question, evidence):
                    self.requests += 1
                    if any(item.chunk.id == 'value' for item in evidence):
                        return Answer(question.id, 'five', 'A', ('value',), False, {})
                    return Answer(question.id, 'unknown', None, (), True,
                                  {'next_read': {'anchor_id': 'anchor', 'before': 0, 'after': 1}})

            argv = ['run_development.py', str(questions), str(database), str(output),
                    '--model', 'offline-fixture', '--top-k', '1', '--followups', '1', '--document-reads']
            with patch.object(sys, 'argv', argv), patch.object(runner, 'SubscriptionGenerator', FakeGenerator), \
                    patch.object(runner.subprocess, 'run') as version:
                version.return_value.stdout = 'fixture-cli'
                runner.main()
            self.assertTrue(configured[0]['allow_document_reads'])
            predictions = json.loads(output.read_text())
            self.assertEqual(predictions['q']['selected_option'], 'A')
            self.assertEqual(predictions['q']['trace']['followup_reads'], 1)
            manifest = json.loads(output.with_suffix('.manifest.json').read_text())
            self.assertTrue(manifest['document_reads'])
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['predictions_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            with zipfile.ZipFile(output.with_suffix('.sources.zip')) as archive:
                self.assertIn(b'--document-reads', archive.read('tools/run_development.py'))
            expected = dict(manifest, document_reads=False)
            with self.assertRaisesRegex(ValueError, 'document_reads'):
                runner.resume_predictions(output, output.with_suffix('.manifest.json'), expected)
