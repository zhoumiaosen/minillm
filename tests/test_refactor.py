"""CPU checks for shared rewards and launcher dispatch."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from trainer.launcher import launch
from trainer.rewards import rep_penalty


class RefactorTests(unittest.TestCase):
    def test_repetition_penalty(self):
        self.assertEqual(rep_penalty(''), 0)
        self.assertEqual(rep_penalty('one two three four'), 0)
        self.assertEqual(rep_penalty('A a a a a'), 0.5)
        self.assertAlmostEqual(rep_penalty('a b a b a b'), 0.5)
        self.assertEqual(rep_penalty('a a a a a', cap=0.2), 0.2)

    def test_dispatch_preserves_arguments_paths_and_exit_status(self):
        root = Path(__file__).resolve().parents[1]
        argv = ['train_rl.py', 'grpo', '--', '--data_path', '/tmp/data with spaces.jsonl']
        with patch.object(sys, 'argv', argv), patch('trainer.launcher.subprocess.run') as run:
            run.return_value.returncode = 7
            with self.assertRaises(SystemExit) as result:
                launch({'grpo': 'train_grpo.py'}, 'test')
        self.assertEqual(result.exception.code, 7)
        self.assertEqual(run.call_args.args[0], [
            sys.executable, str(root / 'trainer/train_grpo.py'),
            '--data_path', '/tmp/data with spaces.jsonl',
        ])
        self.assertEqual(run.call_args.kwargs['cwd'], root / 'trainer')

    def test_invalid_stage_never_launches(self):
        with patch.object(sys, 'argv', ['train_rl.py', 'unknown']), \
                patch('trainer.launcher.subprocess.run') as run:
            with self.assertRaises(SystemExit) as result:
                launch({'ppo': 'train_ppo.py'}, 'test')
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
