from pathlib import Path
import json,hashlib,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from score_development_run import score_run


class DevelopmentScoringTests(unittest.TestCase):
    def test_missing_predictions_remain_in_denominator_and_partial_runs_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);q=root/'dev.questions.jsonl';a=root/'dev.answers.jsonl';p=root/'predictions.json';m=p.with_suffix('.manifest.json')
            q.write_text(''.join(json.dumps({'id':id,'text':'example','options':{'A':'yes'}})+'\n' for id in ('q1','q2')))
            a.write_text(''.join(json.dumps({'question_id':id,'answer':'A'})+'\n' for id in ('q1','q2')))
            (root/'manifest.json').write_text(json.dumps({'files':{f.name:{'path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in (q,a)}}))
            p.write_text(json.dumps({'q1':{'selected_option':'A'}}))
            manifest={'status':'completed','development_only':True,'model':'fixture','question_ids':['q1','q2'],
                      'predictions_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'question_file_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
            m.write_text(json.dumps(manifest))
            result=score_run(p,q,a)
            self.assertEqual(result['score']['accuracy'],.5)
            self.assertEqual(result['score']['failures'],1)
            self.assertFalse(result['goal_achieved'])
            manifest['status']='running';m.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'completed development'):
                score_run(p,q,a)
            manifest['status']='completed';m.write_text(json.dumps(manifest))
            a.write_text(a.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'split checksum'):
                score_run(p,q,a)
