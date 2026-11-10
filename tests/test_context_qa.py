import unittest

from scripts.prepare_context_qa import make_example


class ContextQATest(unittest.TestCase):
    def test_evidence_and_abstention(self):
        qa = {'id': 'a', 'question': 'Code?',
              'answers': [{'text': '7319', 'answer_start': 5}]}
        row = make_example('Code 7319', qa, 'cmrc')
        self.assertEqual(row['conversations'][-1]['content'], '7319')
        self.assertIn('Code 7319', row['conversations'][1]['content'])
        self.assertEqual(row['conversations'][0]['role'], 'system')
        qa['is_impossible'] = True
        qa['answers'] = []
        row = make_example('Nothing here', qa, 'squad')
        self.assertEqual(row['conversations'][-1]['content'], 'Not provided in the context.')
        qa['is_impossible'] = False
        qa['answers'] = [{'text': '7319', 'answer_start': 0}]
        with self.assertRaises(ValueError):
            make_example('Nothing here', qa, 'squad')


if __name__ == '__main__':
    unittest.main()
