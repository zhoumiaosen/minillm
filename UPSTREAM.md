# Source and modifications

This project derives from https://github.com/jingyaogong/minimind at commit
`a3c7b01cc004d5de86aea961f20bf1e638e7c09e`.

The imported source came from the local `minimind/` checkout on 2026-09-10.
The selected tracked code files were unmodified in that checkout. Its local
training scripts and modified ignore rules were not imported. Original file
hashes are recorded in `docs/UPSTREAM-SOURCE.sha256`; that manifest describes
the source before the refactor, not the final contents of modified files.

The Apache License 2.0 is retained verbatim in `LICENSE`. Credit for the original
architecture, tokenizer, dataset loaders, trainers, and utilities belongs to the
MiniMind contributors. No claim of independent original authorship is made.

Changes in this repository:

- Added separate supervised and RL launchers with shared dispatch and stable cwd.
- Extracted identical `rep_penalty` implementations from GRPO, PPO, and agent RL
  into `trainer/rewards.py`.
- Extracted identical mixed-precision setup from GRPO, PPO, DPO, and agent RL
  into `trainer/trainer_utils.py`.
- Fixed GRPO's completed-epoch resume crash discovered during validation:
  track the last processed step so an exhausted loader is a no-op.
- Added a trainer package marker, deferred regression checks, concise setup and
  RL documentation, a core/optional dependency split, and artifact ignore rules.
- Omitted upstream promotional images, long-form READMEs, local training state,
  and Git history. The original dataset documentation remains available.

Modified upstream Python files carry a modification notice. Training objectives,
optimizer loops, model architecture, tokenizer, and checkpoint naming remain
unchanged apart from the documented resume fix. See `docs/VALIDATION.md` for
the tested configurations and remaining validation limits.
