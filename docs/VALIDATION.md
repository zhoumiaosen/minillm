# Runtime validation — 2026-09-11

Bounded checks passed on an NVIDIA RTX 3060 (12 GB), using Python 3.10,
PyTorch 2.6.0+cu124, Transformers 4.57.6, and Datasets 3.6.0.
The original run reported `COMPLETE`; validation outputs are isolated here.

## Results

- Four regression tests passed, including the new GRPO resume regression.
- All eight supervised/RL stage `--help` invocations passed.
- Installed-package consistency check (`python -m pip check`) passed.
- CPU and CUDA precision helper outputs matched the original implementation
  exactly for one deterministic matrix input with float16 and bfloat16.
- The shared repetition helper matched upstream on five sample strings.
- All six saved smoke model state dictionaries contained finite tensors; PPO's
  saved critic was finite too. All six resume files reached epoch index 1, step 2.
- Copied completed pretraining/SFT weights still matched the originals by SHA-256.
- Evaluation of the completed SFT checkpoint passed (8 prompts, up to 24 tokens).
- Evaluation of the new DPO smoke checkpoint passed (8 prompts, up to 8 tokens).

| Run | Result | Seconds |
| --- | --- | --- |
| pretrain | PASS | 28.32 |
| pretrain-resume | PASS | 29.14 |
| sft | PASS | 27.64 |
| sft-resume | PASS | 33.24 |
| dpo | PASS | 29.82 |
| dpo-resume | PASS | 32.7 |
| evaluation | PASS | 16.57 |
| grpo | PASS | 30.51 |
| grpo-resume | PASS | 31.55 |
| ppo | PASS | 44.24 |
| ppo-resume | PASS | 51.72 |
| agent | PASS | 36.08 |
| agent-resume | PASS | 32.76 |

## Scope and inputs

Each training stage used two records, batch size 1, accumulation 1, sequence
length 128, bfloat16, no loader workers, and a save on every step. We ran one
epoch, then resumed with the epoch limit extended to two, exercising both the
exhausted saved epoch and actual subsequent optimization. This is a resume
smoke check, not a recommendation to change a production run's epoch schedule.

Pretraining started from scratch; SFT loaded the completed pretraining weights;
DPO and the other RL trainers loaded the completed SFT weights. Architecture:
dense MiniMind, hidden size 768, 8 layers (63.912M reported trainable parameters).

PPO, GRPO, and agent checks used the real trainers and PyTorch rollout engine,
but replaced the unavailable external reward model with a deterministic
`get_score` test double. Generation was capped at 16 tokens; GRPO/agent used
two generations. GRPO/agent used their default CISPO loss. Agent input used an
empty tool list; real tool-call/multi-turn behavior was not exercised.
DPO used synthetic chosen/rejected pairs and needed no reward model.

The environment is a separate venv populated with copies of the original
installed packages, then the two missing core packages (`einops==0.8.1` and
`sentencepiece==0.2.0`) were installed without changing the original environment.
This is not a fresh dependency-resolution test from an empty environment.

## Fix discovered

GRPO read `step` after an empty loader on completed-epoch resume, raising
`UnboundLocalError`. `tests/test_grpo.py` reproduced the failure before the
fix and passed afterward. Tracking `last_step`, as the other trainers do,
keeps the exhausted epoch a no-op. The original repository was not patched.

## Remaining limits

Real InternLM reward-model loading/scoring, real RL datasets, useful learning,
long training stability, all supported loss variants, gradient accumulation
variants, MoE, distributed training, compilation, SGLang, and full LoRA/
distillation training remain unverified. These checks do not establish model
quality or complete numerical equivalence of training.

## Reproduce and inspect

From this repository, use `.venv/bin/python -m unittest discover -s tests -v`.
Local commands, inputs, harnesses, versions, and complete logs are in
`runs/validation/`, including `results.json`, `rl-results.json`, `unit-final.log`,
`artifacts.log`, and `evaluation-candidate.log`. The RL harness explicitly
labels its reward-model test double. These local artifacts are ignored by Git.
Smoke outputs use `validate_*` prefixes; copied production weights retain their
original names. No GitHub publication was performed.
