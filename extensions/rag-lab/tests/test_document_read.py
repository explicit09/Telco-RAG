import unittest
from raglab.models import Chunk
from raglab.store import SQLiteStore
class DocumentReadTests(unittest.TestCase):
 def test_window_respects_document_corpus_release_and_section(self):
  with SQLiteStore(':memory:') as s:
   def chunk(id,corpus,doc,section,ordinal,release='17'):
    return Chunk(id,corpus,doc,'source','title',section,'paragraph '+id,release,{'ordinal':ordinal})
   s.replace_document('a','doc',[chunk('first','a','doc','one',1),chunk('anchor','a','doc','one',2),chunk('next','a','doc','one',3),chunk('other-section','a','doc','two',4)])
   s.replace_document('b','doc',[chunk('other-corpus','b','doc','one',2)])
   s.replace_document('a','other',[chunk('other-document','a','other','one',2)])
   got=s.read_window('anchor',corpus_ids=['a'],release='17',before=1,after=2)
   self.assertEqual([e.chunk.id for e in got],['first','anchor','next'])
   for scope in ({'corpus_ids':['b'],'release':'17'},{'corpus_ids':['a'],'release':'18'}):
    with self.assertRaises(ValueError):s.read_window('anchor',**scope)
 def test_budgets_and_missing_ordinal_fail_closed(self):
  with SQLiteStore(':memory:') as s:
   s.replace_document('a','d',[Chunk(str(i),'a','d','s','t','one','x'*10,'v',{'ordinal':i}) for i in range(1,8)])
   self.assertEqual(len(s.read_window('4',corpus_ids=['a'],before=3,after=3,limit=2)),2)
   self.assertLessEqual(sum(len(e.chunk.text) for e in s.read_window('4',corpus_ids=['a'],max_chars=19)),19)
   for bad in ({'before':9},{'after':-1},{'limit':17},{'max_chars':24001},{'before':True}):
    with self.assertRaises(ValueError):s.read_window('4',corpus_ids=['a'],**bad)
   s.replace_document('a','bad',[Chunk('bad','a','bad','s','t','one','x','v',{})])
   with self.assertRaises(ValueError):s.read_window('bad',corpus_ids=['a'])

 def test_anchor_is_retained_when_paragraph_has_multiple_pieces(self):
  with SQLiteStore(':memory:') as s:
   s.replace_document('a','d',[Chunk(i,'a','d','s','t','one',i,'v',{'ordinal':1}) for i in ('a','z')])
   self.assertEqual([e.chunk.id for e in s.read_window('z',corpus_ids=['a'],limit=1)],['z'])
