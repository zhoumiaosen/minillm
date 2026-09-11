# RL and preference training

Start with one method and a small dataset. See `VALIDATION.md` for bounded
runtime results. PPO, GRPO, and agent smoke checks use a deterministic test
reward; the real reward-model integration required by the commands below
remains unverified.

## Prerequisites

1. Install this repository's dependencies in its own environment.
2. Place a completed SFT checkpoint in `out/full_sft_768.pth` (default dense
   architecture: hidden size 768, 8 layers). Match all architecture flags when
   using a different checkpoint. Use the bundled, matching tokenizer.
3. Supply the dataset for the selected method, using the formats in `DATA.md`.
4. For GRPO, PPO, and agent RL, supply a compatible reward-model directory.
   Upstream expects InternLM2-1.8B-Reward's custom `get_score(tokenizer, messages)`
   interface, loaded with `trust_remote_code=True`; an arbitrary sequence
   classifier is not interchangeable. Its external model dependencies may need
   additional installation. DPO does not need this model.

Native PyTorch rollouts are the default. All methods load multiple models;
PPO also loads a critic. Small batch sizes alone do not guarantee they fit in
VRAM. Tune memory only after runtime validation is authorized.

## GRPO

```bash
python train_rl.py grpo --data_path /absolute/path/rlaif.jsonl \
  --reward_model_path /absolute/path/internlm2-1_8b-reward \
  --loss_type grpo --batch_size 1 --num_generations 2 \
  --max_seq_len 256 --max_gen_len 128 --num_workers 0
```

Output: `out/grpo_768.pth`. The inherited default loss is CISPO; the command
explicitly selects GRPO clipping. Use at least two generations per prompt for
meaningful group-relative advantages.

## PPO

```bash
python train_rl.py ppo --data_path /absolute/path/rlaif.jsonl \
  --reward_model_path /absolute/path/internlm2-1_8b-reward \
  --batch_size 1 --mini_batch_size 1 \
  --max_seq_len 256 --max_gen_len 128 --num_workers 0
```

Actor output: `out/ppo_actor_768.pth`.

## DPO

```bash
python train_rl.py dpo --data_path /absolute/path/dpo.jsonl \
  --batch_size 1 --max_seq_len 256 --num_workers 0
```

Output: `out/dpo_768.pth`. Supply chosen/rejected conversation pairs.

## Tool-using agent RL

```bash
python train_rl.py agent --data_path /absolute/path/agent_rl.jsonl \
  --reward_model_path /absolute/path/internlm2-1_8b-reward \
  --loss_type grpo --batch_size 1 --num_generations 2 \
  --max_seq_len 256 --max_gen_len 128 --max_total_len 512 --num_workers 0
```

Output: `out/agent_768.pth`. The inherited trainer uses simulated tools and fixed
mock values, not live weather, exchange-rate, or translation services. Its math
tool retains upstream's expression evaluator; this refactor does not make it a
production tool sandbox. Agent RL is an experimental training workflow.

## Resume and evaluate later

Stage resume state lives in `checkpoints/`. Add `--from_resume 1` only to resume
that stage, retaining its data and architecture configuration. The upstream
checkpoint loader uses the default `out/` for starting weights even when
`--save_dir` changes the output destination. Keep defaults for the first run.

From the repository root, after GRPO completes:

```bash
python eval_llm.py --weight grpo --device cuda --max_new_tokens 128
```

Select interactive mode (`1`) or automatic prompts (`0`). Other methods use
their output prefix: `ppo_actor`, `dpo`, or `agent`. This command runs inference.

For distributed training, use the native scripts from `trainer/`, for example
`torchrun --nproc_per_node=2 train_grpo.py ...`; the root launchers are intended
for a single trainer process. Shared checkpoint defaults mean simultaneous
runs of the same stage should use separate repository copies.
