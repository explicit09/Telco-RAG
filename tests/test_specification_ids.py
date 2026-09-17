from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inventory_original_gaps import specification_id


class SpecificationTests(unittest.TestCase):
    def test_versions_parts_and_attachments(self):
        for name,expected in [('23501-ha0.docx','23501'),('22.890-040(cl).docx','22890'),('23700-93-i00.docx','23700-93'),('36101-j50_s00-07.docx','36101'),('29522-ic0_1_Main.docx','29522'),('RS+LDPC.docx',None)]:
            self.assertEqual(specification_id(name),expected)


if __name__=='__main__':unittest.main()
