"""Dispatch a training stage without importing ML libraries in the launcher."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def launch(stages, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('stage', choices=stages)
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER,
                        help='Arguments forwarded to the selected trainer')
    args = parser.parse_args()
    forwarded = args.trainer_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    trainer_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault('HF_HOME', str(trainer_dir.parent / '.cache/huggingface'))
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    env.setdefault('PYTHONUNBUFFERED', '1')
    result = subprocess.run(
        [sys.executable, str(trainer_dir / stages[args.stage]), *forwarded],
        cwd=trainer_dir, env=env,
    )
    raise SystemExit(result.returncode)
