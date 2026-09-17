import unittest
from raglab.models import Chunk,Evidence
from raglab.retrieval import Retriever


class PhraseRetrievalTests(unittest.TestCase):
    def test_phrase_candidates_reach_reranker_with_scope_and_bounded_queries(self):
        bad=Evidence(Chunk('a','manual','a','','','','noise','1'),1)
        good=Evidence(Chunk('b','manual','b','','','','reset timer after five seconds','1'),1)
        class Store:
            calls=[]
            def search(self,query,**kwargs):
                self.calls.append((query,kwargs))
                return [good] if query=='"reset timer"' else [bad]
        class Reranker:
            def rank(self,query,items):
                return sorted(items,key=lambda item:item.chunk.id,reverse=True)
        store=Store()
        query='noise "reset timer" "RESET TIMER" "other phrase" "third phrase"'
        result=Retriever(store,reranker=Reranker(),phrase_search=True).search(query,corpus_ids=['manual'],release='1',limit=1,candidates=1)
        self.assertEqual(result[0].chunk.id,'b')
        self.assertEqual(len(store.calls),3)
        self.assertTrue(all(args=={'corpus_ids':['manual'],'release':'1','limit':1} for _,args in store.calls))
        self.assertIn('phrase',result[0].channel)

    def test_single_phrase_query_is_not_searched_twice(self):
        class Store:
            calls=0
            def search(self,*args,**kwargs):self.calls+=1;return []
        store=Store()
        Retriever(store,phrase_search=True).search('"reset timer"',corpus_ids=['manual'])
        self.assertEqual(store.calls,1)


if __name__=='__main__':unittest.main()
