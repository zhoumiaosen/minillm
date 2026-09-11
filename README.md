# MiniLLM

Train and fine-tune a small language model with separate entry points for
supervised learning, reinforcement learning, and preference optimization.

MiniLLM is a focused refactor of
[jingyaogong/minimind](https://github.com/jingyaogong/minimind). It retains the
upstream model architecture and tokenizer, consolidates shared training helpers,
and adds simpler launchers and regression coverage. The default dense model has
approximately **64 million parameters**.

## What’s included

| Workflow | Entry point | Purpose |
| --- | --- | --- |
| Pretraining | `python train.py pretrain` | Train a model from text |
| Supervised fine-tuning | `python train.py sft` | Train on conversations |
| LoRA | `python train.py lora` | Fine-tune low-rank adapters |
| Distillation | `python train.py distill` | Train from a teacher model |
| DPO | `python train_rl.py dpo` | Optimize chosen/rejected response preferences |
| GRPO / CISPO | `python train_rl.py grpo` | Optimize grouped generated responses |
| PPO | `python train_rl.py ppo` | Train a policy with a critic and rewards |
| Agent RL | `python train_rl.py agent` | Experiment with tool-using agent training |

Also included: checkpoint/resume support, interactive evaluation, model
conversion utilities, and inherited API and web demos.

**Datasets and trained weights are not included in Git.** The tokenizer is
bundled in `model/`.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/zhoumiaosen/minillm.git
cd minillm
python3 -m venv .venv
source .venv/bin/activate
```

Use Python 3.10, the validated version. On systems where virtual-environment creation reports
missing `ensurepip`, install your distribution’s matching Python venv package.

Install a PyTorch build compatible with your hardware and CUDA driver, then
install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The local validation environment used Python 3.10 and PyTorch **2.6.0+cu124** on
an NVIDIA RTX 3060 with 12 GB VRAM. See the
[validation report](docs/VALIDATION.md) for exact versions and test scope.
A fresh installation on another machine has not been verified.

For optional experiment tracking and the API/web utilities:

```bash
python -m pip install -r requirements-extras.txt
```

The inherited `--use_wandb` flag enables **SwanLab**. Native PyTorch rollouts are
the default; the optional SGLang backend requires a separately configured server.

## Prepare your data

```bash
mkdir -p dataset out
```

Place JSONL datasets in `dataset/`, or pass an absolute path with `--data_path`.

| Stage | Default dataset | Starting weights |
| --- | --- | --- |
| Pretrain | `dataset/pretrain_t2t_mini.jsonl` | None |
| SFT | `dataset/sft_t2t_mini.jsonl` | `out/pretrain_768.pth` |
| DPO | `dataset/dpo.jsonl` | `out/full_sft_768.pth` |
| PPO / GRPO | `dataset/rlaif.jsonl` | `out/full_sft_768.pth` |
| Agent RL | `dataset/agent_rl.jsonl` | `out/full_sft_768.pth` |

See [Data formats](docs/DATA.md) for schemas and examples. To use an existing
model, copy its completed weights into `out/` and retain its matching tokenizer.
The defaults expect hidden size 768, 8 layers, and a dense model. Match
`--hidden_size`, `--num_hidden_layers`, and `--use_moe` to other checkpoints.

## Train a model

Run each stage separately, starting with pretraining and then SFT:

```bash
python train.py pretrain \
  --epochs 1 --batch_size 1 --accumulation_steps 1 \
  --max_seq_len 128 --num_workers 0 --device cuda:0 --dtype bfloat16

python train.py sft \
  --epochs 1 --batch_size 1 --accumulation_steps 1 \
  --max_seq_len 128 --num_workers 0 --device cuda:0 --dtype bfloat16
```

These small settings mirror the tested supervised smoke configuration. A smoke
check uses a tiny dataset; these commands process the dataset you supply.
Choose sequence length, batch size, and training duration for your actual task.

Outputs use the stage prefix, such as `out/pretrain_768.pth` and
`out/full_sft_768.pth`. Resume state is stored separately in `checkpoints/`.
To resume an interrupted stage, repeat its command with `--from_resume 1`,
keeping its data, architecture, and training configuration consistent.

**Path behavior:** the launchers run trainers from this repository’s `trainer/`
directory. Built-in defaults resolve correctly; custom relative paths also
resolve from `trainer/`. Use absolute paths for external datasets or models.
Starting weights are loaded from `out/` by default, even if `--save_dir` changes
where new weights are saved.

List stages and inspect their arguments:

```bash
python train.py --help
python train_rl.py --help
python train.py sft --help
```

Stage-specific help imports the ML dependencies. The two top-level help commands
use only the Python standard library.

## Reinforcement learning and preferences

DPO needs preference pairs and an SFT checkpoint, without a separate reward model:

```bash
python train_rl.py dpo \
  --data_path /absolute/path/dpo.jsonl \
  --epochs 1 --batch_size 1 --accumulation_steps 1 \
  --max_seq_len 128 --num_workers 0 --device cuda:0 --dtype bfloat16
```

PPO, GRPO, and agent RL additionally require a compatible reward model. The
inherited implementation expects InternLM2-1.8B-Reward’s custom scoring interface;
other reward models are not automatically interchangeable.

For example, after supplying both the RLAIF dataset and reward model:

```bash
python train_rl.py grpo \
  --data_path /absolute/path/rlaif.jsonl \
  --reward_model_path /absolute/path/internlm2-1_8b-reward \
  --loss_type grpo --batch_size 1 --num_generations 2 \
  --max_seq_len 128 --max_gen_len 16 --num_workers 0
```

The GRPO trainer defaults to **CISPO**; `--loss_type grpo` selects GRPO clipping.
RL loads multiple models, so memory needs differ from supervised training.
See [RL workflows](docs/RL.md) for PPO, agent training, prerequisites, and outputs.

## Chat with a checkpoint

From the repository root:

```bash
python eval_llm.py --weight full_sft --device cuda --max_new_tokens 128
```

Choose `1` for interactive chat or `0` for automatic sample prompts. Change
`--weight` to another output prefix, such as `dpo`, `grpo`, or `ppo_actor`.

## Validation status

The following checks passed on **2026-09-11**:

- Four regression tests, including a fix for GRPO’s completed-epoch resume crash.
- Startup/help checks for all eight training stages.
- Small GPU training and resume runs for pretraining, SFT, and DPO.
- PPO, GRPO/CISPO, and agent training/resume checks with a deterministic test reward.
- Saved-weight finiteness, checkpoint metadata, and CPU/CUDA precision-helper checks.
- Inference using the completed SFT model and a newly generated DPO checkpoint.

**Real reward-model integration remains unverified.** The RL smoke checks used
short synthetic inputs and a test reward; they do not establish training quality.
Full LoRA/distillation training, multi-turn tool use, MoE, distributed execution,
and long-run stability also remain unverified.

Run the regression suite after installing the core dependencies:

```bash
python -m unittest discover -s tests -v
```

Read the [full validation report](docs/VALIDATION.md) for configuration,
evidence, and limitations.

## Project layout

```text
train.py                 Supervised training launcher
train_rl.py              RL and preference-training launcher
trainer/                 Training loops, rollouts, and shared helpers
model/                   Model architecture, LoRA, and tokenizer
dataset/                Dataset loaders and dataset documentation
scripts/                 Conversion, API, and web utilities
tests/                   Regression tests
docs/                    Data, RL, provenance, and validation documentation
eval_llm.py              Evaluation and chat
```

Local environments, datasets, weights, checkpoints, caches, and logs are ignored
by Git. Each checkout has its own default output paths; launchers do not detect
other GPU jobs or coordinate concurrent training runs.

## License and acknowledgments

Licensed under [Apache-2.0](LICENSE). This project builds on the work of the
[MiniMind contributors](https://github.com/jingyaogong/minimind).
See [UPSTREAM.md](UPSTREAM.md) for the source revision, attribution, and changes.
