import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from quarantine_overlap import main


class QuarantineTests(unittest.TestCase):
    def prepare(self,root):
        source=root/'source';source.mkdir();runs=root/'runs';runs.mkdir()
        for name,rows in {
            'dev.questions.jsonl':[{'id':'d1'},{'id':'d2'}],
            'dev.answers.jsonl':[{'question_id':'d1','answer':'A'},{'question_id':'d2','answer':'B'}],
            'test.questions.jsonl':[{'id':'t1'}],
            'test.answers.jsonl':[{'question_id':'t1','answer':'C'}],
        }.items():
            (source/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (source/'manifest.json').write_text('{}')
        audit=root/'audit.json';audit.write_text(json.dumps({'pairs':[{'dev_id':'d1','test_id':'t1'}],'method':'synthetic test'}))
        return source,runs,audit

    def test_held_out_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,runs,audit=self.prepare(root);output=root/'result'
            with patch.object(sys,'argv',['quarantine',str(source),str(audit),str(runs),str(output)]):
                main()
            for name in ('test.questions.jsonl','test.answers.jsonl'):
                self.assertEqual((source/name).read_bytes(),(output/name).read_bytes())
            self.assertEqual(json.loads((output/'dev.questions.jsonl').read_text()),{'id':'d2'})

    def test_previously_used_candidate_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,runs,audit=self.prepare(root)
            (runs/'legacy.json').write_text(json.dumps({'d1':{'selected_option':'A'}}))
            with patch.object(sys,'argv',['quarantine',str(source),str(audit),str(runs),str(root/'result')]):
                with self.assertRaisesRegex(ValueError,'already been used'):
                    main()
