from pathlib import Path
import tempfile
import json
import hashlib
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_development import atomic_json,resume_predictions


class CheckpointTests(unittest.TestCase):
    def test_resume_preserves_predictions_and_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'predictions.json';manifest=Path(folder)/'manifest.json'
            prediction={'q1':{'selected_option':'A'}}
            atomic_json(output,prediction)
            expected={'question_ids':['q1','q2'],'model':'model-a','status':'running'}
            atomic_json(manifest,{**expected,'provider_requests':2,'predictions_sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
            self.assertEqual(resume_predictions(output,manifest,expected),(prediction,2))
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                resume_predictions(output,manifest,{**expected,'model':'model-b'})
            output.write_text('{}')
            with self.assertRaisesRegex(ValueError,'checkpoint hash'):
                resume_predictions(output,manifest,expected)
