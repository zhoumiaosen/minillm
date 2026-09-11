"""Resume regression coverage; requires the core ML dependencies."""

import unittest


class GrpoResumeTests(unittest.TestCase):
    def test_completed_epoch_resume_is_a_noop(self):
        from trainer.train_grpo import grpo_train_epoch

        # A checkpoint saved on the final batch leaves no batches to replay.
        grpo_train_epoch(0, [], 2, None, None, None, start_step=2)


if __name__ == '__main__':
    unittest.main()
