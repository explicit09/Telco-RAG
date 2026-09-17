import unittest
from raglab.followup import answer_with_followup
from raglab.models import Answer, Chunk, Evidence, Question
from raglab.store import SQLiteStore


def evidence(identifier,text,corpus='manual',release='1'):
    return Evidence(Chunk(identifier,corpus,identifier,'source','','',text,release),1)


class FollowupTests(unittest.TestCase):
    def test_fresh_scoped_evidence_recovers_answer_and_records_context(self):
        first=evidence('initial','operating instructions')
        fresh=evidence('fresh','reset timer after five seconds')
        class Generator:
            calls=0
            def generate(self,q,items):
                self.calls+=1
                if any(item.chunk.id=='fresh' for item in items):
                    return Answer(q.id,'five seconds','B',('fresh',),False,{})
                return Answer(q.id,'missing evidence',None,(),True,{'next_search':'"reset timer"'})
        q=Question('q','When does the timer reset?',{'A':'one','B':'five'},('manual',),'1')
        generator=Generator()
        with SQLiteStore(':memory:') as store:
            store.replace_document('manual','fresh',[fresh.chunk])
            answer=answer_with_followup(q,[first],generator,store)
        self.assertEqual(answer.selected_option,'B')
        self.assertEqual(generator.calls,2)
        self.assertEqual(answer.trace['followup_searches'],1)
        self.assertEqual(len(answer.trace['retrieval_rounds']),2)
        self.assertEqual({e['chunk']['id'] for e in answer.trace['retrieval_rounds'][1]['evidence']},{'fresh','initial'})

    def test_budget_and_wrong_corpus_are_enforced(self):
        q=Question('q','question',{},('manual',),'1')
        class Generator:
            calls=0
            def generate(self,q,items):
                self.calls+=1
                return Answer(q.id,'unknown',None,(),True,{'next_search':f'query {self.calls}'})
        class Store:
            calls=0
            def search(self,*args,**kwargs):
                self.calls+=1
                return [evidence(str(self.calls),'new content')]
        generator=Generator();store=Store()
        answer=answer_with_followup(q,[evidence('old','old')],generator,store,max_followups=2)
        self.assertTrue(answer.abstained)
        self.assertEqual((generator.calls,store.calls),(3,2))
        with self.assertRaisesRegex(ValueError,'scope'):
            answer_with_followup(q,[evidence('x','wrong','other')],Generator(),Store())
        class WrongStore:
            def search(self,*args,**kwargs):return [evidence('new','wrong',release='2')]
        with self.assertRaisesRegex(ValueError,'scope'):
            answer_with_followup(q,[evidence('old','old')],Generator(),WrongStore())

    def test_no_new_evidence_stops_without_redundant_generation(self):
        item=evidence('same','same')
        q=Question('q','question',{},('manual',),'1')
        class Generator:
            calls=0
            def generate(self,q,items):
                self.calls+=1
                return Answer(q.id,'unknown',None,(),True,{'next_search':'missing'})
        class Store:
            def search(self,*args,**kwargs):return [item]
        generator=Generator()
        answer_with_followup(q,[item],generator,Store(),max_followups=2)
        self.assertEqual(generator.calls,1)


if __name__=='__main__':unittest.main()
