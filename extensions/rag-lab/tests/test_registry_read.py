import json,tempfile,unittest
from pathlib import Path
from raglab.models import Chunk
from raglab.store import SQLiteStore
from raglab.registry import CorpusRegistry
from raglab.retrieval import Retriever
class RegistryReadTests(unittest.TestCase):
 def test_supplement_owner_and_scope_are_preserved_through_retriever(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d)
   for name,corpus,id in [('base','telecom','base'),('extra','telecom','extra'),('http','http','http')]:
    with SQLiteStore(d/(name+'.sqlite')) as s:
     s.replace_document(corpus,'same-name',[Chunk(id,corpus,'same-name','s','t','section','content','v',{'ordinal':1})])
   p=d/'registry.json';p.write_text(json.dumps({'telecom':['base.sqlite','extra.sqlite'],'http':'http.sqlite'}))
   with CorpusRegistry(p) as registry:
    r=Retriever(registry)
    self.assertEqual([e.chunk.id for e in r.read_window('extra',corpus_ids=['telecom'],release='v')],['extra'])
    self.assertEqual([e.chunk.id for e in r.read_window('http',corpus_ids=['http'],release='v')],['http'])
    for args in ({'corpus_ids':['http'],'release':'v'},{'corpus_ids':['telecom'],'release':'wrong'}):
     with self.assertRaises(ValueError):r.read_window('extra',**args)
 def test_duplicate_anchor_across_parts_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d)
   for name in ('one','two'):
    with SQLiteStore(d/(name+'.sqlite')) as s:s.replace_document('a','d',[Chunk('duplicate','a','d','s','t','sec','text','v',{'ordinal':1})])
   p=d/'registry.json';p.write_text(json.dumps({'a':['one.sqlite','two.sqlite']}))
   with CorpusRegistry(p) as r:
    with self.assertRaises(ValueError):r.read_window('duplicate',corpus_ids=['a'])
