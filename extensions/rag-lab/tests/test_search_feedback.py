import unittest
from raglab.models import Answer, Chunk, Evidence, Question
from raglab.followup import answer_with_followup
from raglab.generation import add_retrieval_feedback, make_payload


class SearchFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.q = Question('q', 'What happens?', {'A': 'notify'}, ('manual',), '1')
        self.anchor = Evidence(Chunk('anchor', 'manual', 'd', '', '', '', 'intro', '1'), 1)

    def test_empty_search_can_be_reformulated_within_shared_budget(self):
        anchor = self.anchor
        class Store:
            calls = []
            def search(s, query, **kwargs):
                s.calls.append((query, kwargs))
                return [] if kwargs.get('match_mode') == 'all' else [Evidence(Chunk('rule', 'manual', 'd', '', '', '', 'notify', '1'), 1)]
        class Generator:
            calls = 0
            feedbacks = []
            def generate(g, q, evidence, *, retrieval_feedback=None):
                g.calls += 1; g.feedbacks.append(retrieval_feedback)
                if any(e.chunk.id == 'rule' for e in evidence):
                    return Answer(q.id, 'notify', 'A', ('rule',), False, {})
                mode = 'any' if retrieval_feedback else 'all'
                return Answer(q.id, 'unknown', None, (), True, {'next_search': 'condition topic', 'next_search_mode': mode})
        store, generator = Store(), Generator()
        answer = answer_with_followup(self.q, [anchor], generator, store, max_followups=2, allow_search_modes=True)
        self.assertEqual(answer.selected_option, 'A')
        self.assertEqual(generator.calls, 3)
        self.assertEqual(len(store.calls), 2)
        self.assertEqual(generator.feedbacks[1]['result_count'], 0)
        self.assertIsNone(generator.feedbacks[2])
        self.assertEqual(answer.trace['retrieval_rounds'][1]['retrieval_feedback']['match_mode'], 'all')

    def test_same_query_and_mode_are_not_executed_twice(self):
        class Store:
            calls = 0
            def search(s, *args, **kwargs):
                s.calls += 1
                return []
        class Generator:
            calls = 0
            def generate(g, q, evidence, *, retrieval_feedback=None):
                g.calls += 1
                return Answer(q.id, 'unknown', None, (), True, {'next_search': 'condition', 'next_search_mode': 'all'})
        s, g = Store(), Generator()
        answer = answer_with_followup(self.q, [self.anchor], g, s, max_followups=2, allow_search_modes=True)
        self.assertTrue(answer.abstained)
        self.assertEqual(s.calls, 1)
        self.assertEqual(g.calls, 2)

    def test_feedback_is_separate_from_public_question_and_validated(self):
        payload = make_payload(self.q, [self.anchor], model='fixture', max_output_tokens=100)
        original = payload['input'][1]['content']
        feedback = dict(query='condition', match_mode='all', result_count=0, new_evidence_count=0, status='no_new_evidence')
        add_retrieval_feedback(payload, feedback)
        self.assertEqual(payload['input'][1]['content'], original)
        self.assertIn('retrieval_feedback', payload['input'][2]['content'])
        with self.assertRaises(ValueError):
            add_retrieval_feedback(payload, dict(feedback, answer='A'))
