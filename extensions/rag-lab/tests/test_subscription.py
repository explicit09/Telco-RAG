import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from raglab.models import Question, Chunk, Evidence
from raglab.subscription import SubscriptionGenerator


class SubscriptionTests(unittest.TestCase):
    def setUp(self):
        self.question = Question('dev', 'Which?', {'A':'30 seconds'}, ('demo',))
        self.evidence = [Evidence(Chunk('c','demo','d','s','t','','30 seconds'), 1)]

    def process(self, events):
        def run(command, **kwargs):
            self.assertNotIn('OPENAI_API_KEY', kwargs['env'])
            self.assertIn('--ignore-user-config', command)
            output = Path(command[command.index('--output-last-message') + 1])
            output.write_text(json.dumps(dict(text='30 seconds', selected_option='A', abstained=False,
                citations=['c'], quotes=[dict(chunk_id='c',quote='30 seconds')],next_search=None)))
            return SimpleNamespace(returncode=0, stdout='\n'.join(json.dumps(e) for e in events))
        return run

    def test_structured_answer_and_request_budget(self):
        generator = SubscriptionGenerator(max_requests=1)
        events = [{'type':'item.completed','item':{'type':'agent_message'}}, {'type':'turn.completed','usage':{}}]
        with patch('subprocess.run', side_effect=self.process(events)):
            self.assertEqual(generator.generate(self.question,self.evidence).selected_option,'A')
            with self.assertRaises(RuntimeError):
                generator.generate(self.question,self.evidence)

    def test_tool_activity_rejected_even_if_answer_exists(self):
        events=[{'type':'item.completed','item':{'type':'command_execution'}}]
        with patch('subprocess.run', side_effect=self.process(events)):
            with self.assertRaisesRegex(RuntimeError,'Non-answer event'):
                SubscriptionGenerator().generate(self.question,self.evidence)

    def test_held_out_mode_refused(self):
        with self.assertRaises(ValueError):
            SubscriptionGenerator(development_only=False)
