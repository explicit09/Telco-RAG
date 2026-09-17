from pathlib import Path
import hashlib,subprocess,sys,tempfile,unittest,zipfile
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_word_sources import prepare,repair_orphan_revision_closings


class WordPreparationTests(unittest.TestCase):
    @unittest.skipUnless(Path('/usr/bin/textutil').is_file(),'macOS converter required')
    def test_real_legacy_doc_conversion_retains_fixture_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);text=root/'fixture.txt';doc=root/'fixture.doc';target=root/'prepared.docx'
            text.write_text('Timer reset requires five seconds.\nSecond paragraph remains visible.')
            subprocess.run(['/usr/bin/textutil','-convert','doc','-output',str(doc),str(text)],check=True,capture_output=True)
            metadata=prepare(doc,target)
            self.assertIn('converter_sha256',metadata)
            with zipfile.ZipFile(target) as archive:
                xml=ET.fromstring(archive.read('word/document.xml'))
                content=''.join(xml.itertext())
            self.assertIn('Timer reset requires five seconds.',content)
            self.assertIn('Second paragraph remains visible.',content)

    def test_repairs_only_orphan_revision_closing_without_changing_text(self):
        original=b'<w:document xmlns:w="urn:word"><w:p><w:ins><w:t>kept</w:t></w:ins></w:ins></w:p></w:document>'
        repaired,count=repair_orphan_revision_closings(original)
        self.assertEqual(count,1)
        self.assertEqual(repaired,original.replace(b'</w:ins></w:ins>',b'</w:ins>'))
        self.assertEqual(''.join(ET.fromstring(repaired).itertext()),'kept')
        with self.assertRaises(ValueError):repair_orphan_revision_closings(b'<a><b>text</a>')
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.docx';target=Path(folder)/'prepared.docx'
            with zipfile.ZipFile(source,'w') as z:z.writestr('word/document.xml',original)
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(prepare(source,target)['removed_tags'],1)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
            with zipfile.ZipFile(target) as z:self.assertEqual(z.read('word/document.xml'),repaired)


if __name__=='__main__':unittest.main()
