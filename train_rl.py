"""Separate entry point for reinforcement learning and preference optimization."""

from trainer.launcher import launch


if __name__ == '__main__':
    launch({
        'grpo': 'train_grpo.py',
        'ppo': 'train_ppo.py',
        'dpo': 'train_dpo.py',
        'agent': 'train_agent.py',
    }, 'MiniLLM RL and preference training')
