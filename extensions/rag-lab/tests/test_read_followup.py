import unittest
from raglab.models import Answer,Chunk,Evidence,Question
from raglab.store import SQLiteStore
from raglab.generation import make_payload,make_read_payload,parse_read_answer
from raglab.followup import answer_with_followup
class ReadFollowupTests(unittest.TestCase):
 def setUp(self):
  self.q=Question('q','What value?',{'A':'five'},('manual',),'1')
  self.anchor=Evidence(Chunk('anchor','manual','doc','s','t','sec','The value is given next.','1',{'ordinal':1}),1)
 def response(self,request):return {'text':'unknown','selected_option':None,'abstained':True,'citations':[],'quotes':[],'next_search':None,'next_read':request}
 def test_schema_is_opt_in_and_anchor_validation_is_strict(self):
  enhanced=make_read_payload(self.q,[self.anchor],model='m',max_output_tokens=10)
  base=make_payload(self.q,[self.anchor],model='m',max_output_tokens=10)
  self.assertIn('next_read',enhanced['text']['format']['schema']['required'])
  self.assertEqual(enhanced['text']['format']['schema']['properties']['selected_option']['enum'], ['A', None])
  self.assertNotIn('next_read',base['text']['format']['schema']['required'])
  for request in [{'anchor_id':'unseen','before':0,'after':1},{'anchor_id':'anchor','before':True,'after':1},{'anchor_id':'anchor','before':0,'after':9}]:
   with self.assertRaises(ValueError):parse_read_answer(self.q,[self.anchor],self.response(request))
  data=self.response({'anchor_id':'anchor','before':0,'after':1});data['next_search']='query'
  with self.assertRaises(ValueError):parse_read_answer(self.q,[self.anchor],data)
 def test_real_store_read_recovers_answer_with_shared_budget(self):
  class Generator:
   calls=0
   def generate(g,q,evidence):
    g.calls+=1
    if any(e.chunk.id=='value' for e in evidence):return Answer(q.id,'five','A',('value',),False,{})
    return Answer(q.id,'unknown',None,(),True,{'next_read':{'anchor_id':'anchor','before':0,'after':1},'next_search':None})
  with SQLiteStore(':memory:') as store:
   store.replace_document('manual','doc',[self.anchor.chunk,Chunk('value','manual','doc','s','t','sec','The value is five.','1',{'ordinal':2})])
   g=Generator();a=answer_with_followup(self.q,[self.anchor],g,store,allow_document_reads=True)
   self.assertEqual(a.selected_option,'A');self.assertEqual(g.calls,2)
   self.assertEqual(a.trace['followup_reads'],1);self.assertEqual(a.trace['followup_searches'],0)
   with self.assertRaises(ValueError):answer_with_followup(self.q,[self.anchor],Generator(),store)
   g=Generator();a=answer_with_followup(self.q,[self.anchor],g,store,allow_document_reads=True,max_followups=0)
   self.assertTrue(a.abstained);self.assertEqual(g.calls,1)
 def test_unseen_read_anchor_is_rejected_before_store_access(self):
  class Generator:
   def generate(g,q,evidence):return Answer(q.id,'unknown',None,(),True,{'next_read':{'anchor_id':'unseen','before':0,'after':1}})
  class Store:
   def read_window(s,*args,**kwargs):raise AssertionError('must not be called')
  with self.assertRaisesRegex(ValueError,'anchor'):answer_with_followup(self.q,[self.anchor],Generator(),Store(),allow_document_reads=True)

 def test_read_and_search_share_the_same_followup_budget(self):
  requests=[{'next_read':{'anchor_id':'anchor','before':0,'after':1}}, {'next_search':'missing value'}, {'next_search':'another query'}]
  class Generator:
   calls=0
   def generate(g,q,evidence):
    trace=requests[g.calls];g.calls+=1
    return Answer(q.id,'unknown',None,(),True,trace)
  class Store:
   calls=[]
   def read_window(s,*args,**kwargs):
    s.calls.append('read')
    return [Evidence(Chunk('neighbor','manual','doc','s','t','sec','More context.','1',{'ordinal':2}),1)]
   def search(s,*args,**kwargs):
    s.calls.append('search')
    return [Evidence(Chunk('result','manual','other','s','t','sec','Other context.','1',{'ordinal':1}),1)]
  g=Generator();s=Store()
  a=answer_with_followup(self.q,[self.anchor],g,s,allow_document_reads=True,max_followups=2)
  self.assertEqual(s.calls,['read','search'])
  self.assertEqual(g.calls,3)
  self.assertEqual(a.trace['followup_reads'],1)
  self.assertEqual(a.trace['followup_searches'],1)

 def test_out_of_scope_read_results_never_reach_the_generator(self):
  for corpus,release in [('other','1'),('manual','2')]:
   class Generator:
    calls=0
    def generate(g,q,evidence):
     g.calls+=1
     return Answer(q.id,'unknown',None,(),True,{'next_read':{'anchor_id':'anchor','before':0,'after':1}})
   class Store:
    def read_window(s,*args,**kwargs):
     return [Evidence(Chunk('foreign',corpus,'doc','s','t','sec','Foreign context.',release,{'ordinal':2}),1)]
   g=Generator()
   with self.assertRaisesRegex(ValueError,'scope'):
    answer_with_followup(self.q,[self.anchor],g,Store(),allow_document_reads=True)
   self.assertEqual(g.calls,1)
