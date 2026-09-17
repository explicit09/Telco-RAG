import unittest
from raglab.generation import ANSWER_SCHEMA, make_payload
from raglab.models import Question


class AnswerSchemaTests(unittest.TestCase):
    def test_option_constraints_are_local_to_each_question(self):
        first = make_payload(Question('a', 'Which?', {'option 1': 'one', 'option 2': 'two'}), [], model='fixture', max_output_tokens=100)
        second = make_payload(Question('b', 'Which?', {'B': 'another'}), [], model='fixture', max_output_tokens=100)
        open_ended = make_payload(Question('c', 'Explain'), [], model='fixture', max_output_tokens=100)
        def allowed(payload):
            return payload['text']['format']['schema']['properties']['selected_option']['enum']
        self.assertEqual(allowed(first), ['option 1', 'option 2', None])
        self.assertEqual(allowed(second), ['B', None])
        self.assertEqual(allowed(open_ended), [None])
        self.assertNotIn('enum', ANSWER_SCHEMA['properties']['selected_option'])
        self.assertEqual(allowed(first), ['option 1', 'option 2', None])
