import tempfile,unittest
from pathlib import Path
from zipfile import ZipFile
from raglab.ingest import ingest_file
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
class WordMathTests(unittest.TestCase):
 def ingest(self,body):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'sample.docx'
   with ZipFile(p,'w') as z:z.writestr('word/document.xml',f'<w:document xmlns:w="{W}" xmlns:m="{M}"><w:body>{body}</w:body></w:document>')
   return ingest_file(p,corpus_id='manual')
 def test_math_only_equation_is_not_dropped(self):
  c=self.ingest('<w:p><m:oMath><m:sSubSup><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:t>i</m:t></m:r></m:sub><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSubSup><m:r><m:t>=1</m:t></m:r></m:oMath></w:p>')
  self.assertEqual(c[0].text,'(P)_{i}^{2}=1')
 def test_fraction_survives_table_ingestion(self):
  c=self.ingest('<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Ratio</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><m:oMath><m:f><m:num><m:r><m:t>a+b</m:t></m:r></m:num><m:den><m:r><m:t>c</m:t></m:r></m:den></m:f></m:oMath></w:p></w:tc></w:tr></w:tbl>')
  self.assertIn('Ratio | ((a+b)/(c))',c[1].text)
 def test_unknown_structure_is_explicit(self):
  c=self.ingest('<w:p><m:oMath><m:future><m:r><m:t>x</m:t></m:r></m:future></m:oMath></w:p>')
  self.assertIn('[unsupported future: x]',c[0].text)
