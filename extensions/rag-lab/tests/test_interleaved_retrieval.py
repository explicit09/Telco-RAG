import unittest
from dataclasses import replace
from raglab.models import Chunk, Evidence
from raglab.retrieval import Retriever


class InterleavedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.items = [Evidence(Chunk(str(i), 'manual', 'doc', '', '', '', str(i), '1'), 1) for i in range(8)]
        items = self.items

        class Store:
            def search(self, query, **kwargs):
                return items[:kwargs['limit']]

        class Ranker:
            def rank(self, query, evidence):
                return list(reversed(evidence))

        self.store, self.ranker = Store(), Ranker()

    def test_disagreement_preserves_both_orders_with_fixed_budget(self):
        retrieval = Retriever(self.store, reranker=self.ranker, rerank_strategy='interleave')
        result = retrieval.search('value', corpus_ids=['manual'], release='1', limit=4)
        self.assertEqual([e.chunk.id for e in result], ['0', '7', '1', '6'])
        default = Retriever(self.store, reranker=self.ranker).search('value', corpus_ids=['manual'], limit=4)
        self.assertEqual([e.chunk.id for e in default], ['7', '6', '5', '4'])

    def test_overlapping_orders_do_not_waste_passage_slots(self):
        class Ranker:
            def rank(self, query, evidence):
                return evidence
        retrieval = Retriever(self.store, reranker=Ranker(), rerank_strategy='interleave')
        result = retrieval.search('value', corpus_ids=['manual'], limit=8)
        self.assertEqual([e.chunk.id for e in result], [str(i) for i in range(8)])

    def test_interleave_still_rejects_reranker_injection(self):
        class Ranker:
            def rank(self, query, evidence):
                return [replace(evidence[0], chunk=replace(evidence[0].chunk, corpus_id='foreign'))]
        with self.assertRaisesRegex(ValueError, 'outside candidate'):
            Retriever(self.store, reranker=Ranker(), rerank_strategy='interleave').search('value', corpus_ids=['manual'])
        with self.assertRaises(ValueError):
            Retriever(self.store, rerank_strategy='unknown')
