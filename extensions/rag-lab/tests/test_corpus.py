from pathlib import Path
import unittest
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from raglab.ingest import ingest_file
from raglab.store import SQLiteStore


class CorpusTests(unittest.TestCase):
  def test_corpora_and_releases_are_isolated(self):
    with self.subTest():
      tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("shared term from first")
    second.write_text("shared term from second")
    with SQLiteStore(tmp_path / "db.sqlite") as store:
        store.replace_document("a", "doc", ingest_file(first, corpus_id="a", document_id="doc", release="v1"))
        store.replace_document("b", "doc", ingest_file(second, corpus_id="b", document_id="doc", release="v2"))
        assert {e.chunk.corpus_id for e in store.search("shared", corpus_ids=["a"])} == {"a"}
        assert store.search("shared", corpus_ids=["a"], release="v2") == []
        assert len(store.chunks(["b"], release="v2")) == 1


  def test_replacement_removes_stale_chunks(self):
    tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
    source = tmp_path / "doc.txt"
    source.write_text("old unique text")
    with SQLiteStore(tmp_path / "db.sqlite") as store:
        store.replace_document("a", "doc", ingest_file(source, corpus_id="a", release="v1"))
        source.write_text("new unique text")
        store.replace_document("a", "doc", ingest_file(source, corpus_id="a", release="v2"))
        assert store.search("old", corpus_ids=["a"]) == []
        assert store.search("new", corpus_ids=["a"], release="v2")


  def test_malformed_and_unsupported_files_rejected(self):
    tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json")
    with self.assertRaises(ValueError):
        ingest_file(bad, corpus_id="a")
    with self.assertRaises(ValueError):
        ingest_file(tmp_path / "unknown.bin", corpus_id="a")


  def test_docx_table_rows_and_headers_are_preserved(self):
    tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
    docx = tmp_path / "table.docx"
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="{ns}"><w:body>
      <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Results</w:t></w:r></w:p>
      <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    <w:sectPr/></w:body></w:document>'''
    with ZipFile(docx, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
    chunks = ingest_file(docx, corpus_id="a")
    self.assertEqual(len(chunks), 2)
    self.assertIn("Name | Value", chunks[1].text)
    self.assertEqual(chunks[1].metadata["source_kind"], "table")
    self.assertEqual(chunks[1].section, "Results")

  def test_repeated_paragraphs_and_natural_query(self):
    folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
    source = folder / "manual.txt"
    source.write_text("The timer expires after thirty seconds.\n\nThe timer expires after thirty seconds.")
    chunks = ingest_file(source, corpus_id="manual", document_id="timer")
    self.assertEqual(len({c.id for c in chunks}), 2)
    with SQLiteStore(":memory:") as store:
        store.replace_document("manual", "timer", chunks)
        self.assertTrue(store.search("When does this timer expire?", corpus_ids=["manual"]))

  def test_empty_document_rejected(self):
    folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
    source = folder / "empty.txt"
    source.write_text("")
    with self.assertRaises(ValueError):
        ingest_file(source, corpus_id="manual")

  def test_wrapped_text_preserves_whole_paragraph(self):
    folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
    source = folder / "wrapped.txt"
    source.write_text("The server MUST NOT\nsend content in a HEAD response.\n\nA different paragraph.")
    chunks = ingest_file(source, corpus_id="http")
    self.assertEqual(len(chunks), 2)
    self.assertIn("MUST NOT send content", chunks[0].text)
