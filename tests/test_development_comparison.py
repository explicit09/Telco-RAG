from pathlib import Path
import hashlib,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_development_runs import compare


class ComparisonTests(unittest.TestCase):
    def test_paired_outcomes_and_mismatched_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);q=root/'dev.questions.jsonl';a=root/'dev.answers.jsonl'
            q.write_text(''.join(json.dumps({'id':i,'text':'example','options':{'A':'yes','B':'no'}})+'\n' for i in ('q1','q2','q3','q4')))
            a.write_text(''.join(json.dumps({'question_id':i,'answer':'A'})+'\n' for i in ('q1','q2','q3','q4')))
            (root/'manifest.json').write_text(json.dumps({'files':{f.name:{'path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in (q,a)}}))
            paths=[root/'left.json',root/'right.json']
            for path,answers in zip(paths,(['A','B','A',None],['A','A','B',None])):
                path.write_text(json.dumps({f'q{n}':{'selected_option':answer,'abstained':answer is None} for n,answer in enumerate(answers,1)}))
                path.with_suffix('.manifest.json').write_text(json.dumps({'status':'completed','development_only':True,'model':'fixture',
                    'question_ids':['q1','q2','q3','q4'],'predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'question_file_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'corpus_manifest':{'a':{'source_manifest_sha256':'same','seconds':1 if path==paths[0] else 2}}}))
            report=compare(*paths,q,a)
            self.assertEqual(report['paired_counts'],dict(both_correct=1,right_improved=1,right_regressed=1,both_incorrect=1))
            self.assertEqual(report['accuracy_difference'],0)
            self.assertFalse(report['goal_achieved'])
            manifest=paths[1].with_suffix('.manifest.json');data=json.loads(manifest.read_text());data['model']='changed';manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'model'):
                compare(*paths,q,a)


if __name__=='__main__':
    unittest.main()
