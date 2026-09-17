import unittest
from raglab.models import Answer, Chunk, Evidence, Question
from raglab.generation import make_payload, make_search_payload, parse_search_answer
from raglab.followup import answer_with_followup
from raglab.store import SQLiteStore


class SearchActionTests(unittest.TestCase):
    def setUp(self):
        self.q = Question('q', 'What happens?', {'A': 'notify'}, ('manual',), '1')
        self.e = [Evidence(Chunk('anchor', 'manual', 'doc', '', '', '', 'See service requirements.', '1'), 1)]
        self.response = dict(text='unknown', selected_option=None, abstained=True, citations=[], quotes=[],
                             next_search='"cannot provide" QoS', next_search_mode='all')

    def test_mode_schema_is_opt_in_and_preserves_option_constraints(self):
        payload = make_search_payload(self.q, self.e, model='fixture', max_output_tokens=100, allow_document_reads=True)
        schema = payload['text']['format']['schema']
        self.assertIn('next_search_mode', schema['required'])
        self.assertIn('next_read', schema['required'])
        self.assertEqual(schema['properties']['selected_option']['enum'], ['A', None])
        plain = make_payload(self.q, self.e, model='fixture', max_output_tokens=100)
        self.assertNotIn('next_search_mode', plain['text']['format']['schema']['required'])
        answer = parse_search_answer(self.q, self.e, dict(self.response, next_read=None), allow_document_reads=True)
        self.assertEqual(answer.trace['next_search_mode'], 'all')

    def test_invalid_or_unrequested_modes_are_rejected(self):
        for change in [dict(next_search_mode='bad'), dict(next_search=None), dict(next_search_mode=None), dict(abstained=False)]:
            with self.assertRaises(ValueError):
                parse_search_answer(self.q, self.e, dict(self.response, **change))

    def test_all_mode_reaches_store_with_the_existing_call_budget(self):
        class Generator:
            calls = 0
            def generate(g, q, evidence):
                g.calls += 1
                if any(e.chunk.id == 'rule' for e in evidence):
                    return Answer(q.id, 'notify', 'A', ('rule',), False, {})
                return Answer(q.id, 'unknown', None, (), True,
                              {'next_search': '"cannot provide" QoS', 'next_search_mode': 'all'})
        with SQLiteStore(':memory:') as store:
            store.replace_document('manual', 'rules', [Chunk('rule', 'manual', 'rules', '', '', '', 'If we cannot provide QoS, notify.', '1')])
            g = Generator()
            answer = answer_with_followup(self.q, self.e, g, store, allow_search_modes=True, max_followups=1)
            self.assertEqual(answer.selected_option, 'A')
            self.assertEqual(g.calls, 2)
            self.assertEqual(answer.trace['followup_searches'], 1)
            with self.assertRaisesRegex(ValueError, 'disabled'):
                answer_with_followup(self.q, self.e, Generator(), store)
