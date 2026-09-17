from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from freeze_development_sample import select


class SampleTests(unittest.TestCase):
    def test_exclusion_release_coverage_and_order_independence(self):
        rows=[{'id':f'q{i}','release':str(i%5)} for i in range(40)]
        excluded={'q0','q1','q2'}
        chosen=select(rows,excluded,15,42)
        self.assertEqual(len(chosen),15)
        self.assertEqual({q['release'] for q in chosen},{str(i) for i in range(5)})
        self.assertFalse({q['id'] for q in chosen}&excluded)
        self.assertEqual(chosen,select(list(reversed(rows)),excluded,15,42))
        with self.assertRaisesRegex(ValueError,'cover every release'):
            select(rows,excluded,3,42)


if __name__=='__main__':
    unittest.main()
