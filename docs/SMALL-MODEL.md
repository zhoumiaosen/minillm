# Smaller MiniLLM: 17M pilot

**Result (2026-09-11): size and execution checks passed; quality acceptance failed.**
Keep the 64M model as the current reference. The smaller weights are experimental
training artifacts, not a replacement release or a conversion of the existing model.

## Architecture and measured size

The student uses the existing dense architecture with hidden width 384, 8 layers,
8 query heads, 4 KV heads, head dimension 48, intermediate width 1216, and the
same 6400-token tokenizer. Input/output embeddings remain tied. No architecture
code or default model size changed.

It has **17,210,496 unique parameters**, versus **63,912,192** in the teacher:
**73.1% fewer parameters**. Serialized weight files are FP16; their size includes
serialization and the existing writer's separately converted tied-weight entries.
MiB means 1,048,576 bytes; file size is not training VRAM consumption.

| Checkpoint | Parameters | File bytes | File MiB | Assistant-token NLL |
| --- | ---: | ---: | ---: | ---: |
| teacher | 63,912,192 | 137,684,000 | 131.31 | 1.125190 |
| sft | 17,210,496 | 39,365,466 | 37.54 | 5.705477 |
| distill | 17,210,496 | 39,365,910 | 37.54 | 5.315734 |

The distilled weight file is 71.4% smaller than the teacher file.

## Training budget and execution

Hardware: RTX 3060, 12 GB VRAM. Environment: Python 3.10, PyTorch 2.6.0+cu124,
Transformers 4.57.6, Datasets 3.6.0. No dependencies were added for this pilot.
Use the Python environment configured in the main README.

| Stage | Records | Epochs | Batch | Sequence limit | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrain from scratch | 2048 | 1 | 4 | 256 | 45.75 |
| Student SFT | 1024 | 1 | 4 | 256 | 32.01 |
| Distillation from the 64M teacher | 1024 | 1 | 2 | 256 | 66.85 |

All stages used bfloat16, accumulation 1, zero data-loader workers, and a
20-minute per-stage time limit. Times include startup and checkpoint writes;
they are local observations, not training-time promises. The teacher remained
in eval mode with gradients disabled; teacher forward passes used no-grad.
The teacher must explicitly use `--teacher_use_moe 0` because the inherited
trainer's teacher default is MoE.

Source data: `pretrain_t2t_mini.jsonl` and `sft_t2t_mini.jsonl`. Selection retained
the first unique records after canonical JSON deduplication. The first 1024 of
1088 SFT records were used for student SFT/distillation; the remaining 64 were
held out from those two stages. Two training records were used for smoke tests.
All selected SFT records passed a nonempty-supervision check at the configured
sequence limits. Data and full source hashes remain in local `runs/smaller/`.

## Quality comparison and decision

All three models were evaluated on the same tokenized inputs and assistant-label
masks: **12,517 supervised tokens** over 64 records, sequence limit 256. Evaluation
used strict checkpoint loading and token-weighted cross-entropy (NLL, nats/token;
lower is better). Input/mask SHA-256 matched across all three evaluations:
`6752f28870abeab75f880cb1d953ed9e3e62540c51a82d4f2d06562cd7fb8bb9`.

The provisional gate was NLL <= 1.25 times teacher NLL, or **1.406487**.
The SFT student scored **5.705477** and the distilled student **5.315734**.
Both failed. Distillation lowered student NLL by
6.8%, but that metric improvement does not make the model ready for use.

Eight fixed prompts were reviewed, with generation capped at 128 new tokens.
Both students produced repetitive, fragmented text and failed to give coherent
answers, including to the sky-color and Python Fibonacci prompts. Distillation
did not resolve that behavior. The teacher was more fluent but also produced
factual and coding errors; it is a comparison reference, not ground truth.

The teacher had already trained on the source SFT corpus. These 64 examples are
student-held-out, not an unbiased generalization benchmark for the teacher.
Cross-corpus pretraining overlap was not excluded. Sampling in the existing
chat evaluator is nondeterministic; its speed output was not used as a benchmark.

**Decision:** retain both student checkpoints for experiments; promote neither.
A separately approved, larger data/training budget is the next experiment. More
training may improve quality, but its outcome is not established by this pilot.
A 512-wide, 8-layer alternative has about 30M parameters and remains untested.

## Reproduce the stages

Provide a completed dense teacher at `out/full_sft_768.pth` and prepare the pilot
JSONL splits described above in `runs/smaller/`. Model weights and datasets are
local inputs, not included in Git. Run from the repository root, sequentially.
The launchers resolve custom relative paths from `trainer/`.

```bash
timeout --signal=INT 20m python train.py pretrain --hidden_size 384 --num_hidden_layers 8 --use_moe 0 --from_weight none --save_weight small384l8_pretrain --data_path ../runs/smaller/pretrain.jsonl --epochs 1 --batch_size 4 --accumulation_steps 1 --max_seq_len 256 --num_workers 0 --device cuda:0 --dtype bfloat16 --log_interval 20 --save_interval 128
```

```bash
timeout --signal=INT 20m python train.py sft --hidden_size 384 --num_hidden_layers 8 --use_moe 0 --from_weight small384l8_pretrain --save_weight small384l8_sft --data_path ../runs/smaller/train.jsonl --epochs 1 --batch_size 4 --accumulation_steps 1 --max_seq_len 256 --num_workers 0 --device cuda:0 --dtype bfloat16 --log_interval 20 --save_interval 64
```

```bash
timeout --signal=INT 20m python train.py distill --student_hidden_size 384 --student_num_layers 8 --student_use_moe 0 --teacher_hidden_size 768 --teacher_num_layers 8 --teacher_use_moe 0 --from_student_weight small384l8_sft --from_teacher_weight full_sft --save_weight small384l8_distill --data_path ../runs/smaller/train.jsonl --epochs 1 --batch_size 2 --accumulation_steps 1 --max_seq_len 256 --num_workers 0 --device cuda:0 --dtype bfloat16 --alpha 0.5 --temperature 1.5 --log_interval 20 --save_interval 64
```

Outputs:

- `out/small384l8_pretrain_384.pth`
- `out/small384l8_sft_384.pth`
- `out/small384l8_distill_384.pth`

Their optimizer/resume files live in `checkpoints/` with the same prefixes and
`_resume.pth` suffix. The layer count is not encoded by the generic checkpoint
writer, so the `384l8` prefix distinguishes this configuration. Always pass
`--hidden_size 384 --num_hidden_layers 8` when evaluating these weights.

## Reproduce evaluation

For the quantitative comparison, save the following as the ignored local file
`runs/smaller/evaluate.py`:

```python
import json
import hashlib
from pathlib import Path
import random
import sys
import torch
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

weight, width = Path(sys.argv[1]), int(sys.argv[2])
config = MiniMindConfig(hidden_size=width, num_hidden_layers=8, use_moe=False)
model = MiniMindForCausalLM(config)
model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True),
                      strict=True)
model.eval().requires_grad_(False).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(root / 'model')
# Reset Python randomness after construction so SFT augmentation matches.
random.seed(42)
torch.manual_seed(42)
data = SFTDataset(str(root / 'runs/smaller/holdout.jsonl'), tokenizer, max_length=256)
total_loss, total_tokens = 0.0, 0
input_hash = hashlib.sha256()
for i in range(len(data)):
    ids, labels = data[i]
    input_hash.update(ids.numpy().tobytes())
    input_hash.update(labels.numpy().tobytes())
    count = int((labels[1:] != -100).sum())
    if not count:
        raise ValueError(f'No supervised tokens in held-out record {i}')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(ids.unsqueeze(0).cuda(), labels=labels.unsqueeze(0).cuda())
    if not torch.isfinite(result.loss):
        raise ValueError(f'Nonfinite loss at record {i}')
    total_loss += result.loss.item() * count
    total_tokens += count
print(json.dumps({
    'weight': str(weight), 'hidden_size': width, 'layers': 8,
    'parameters': sum(p.numel() for p in model.parameters()),
    'file_bytes': weight.stat().st_size,
    'assistant_tokens': total_tokens, 'assistant_nll': total_loss / total_tokens,
    'input_sha256': input_hash.hexdigest(),
}, indent=2))
```

Then run:

```bash
python runs/smaller/evaluate.py out/full_sft_768.pth 768
python runs/smaller/evaluate.py out/small384l8_sft_384.pth 384
python runs/smaller/evaluate.py out/small384l8_distill_384.pth 384
```

For the same eight automatic chat prompts:

```bash
printf '0\n' | python eval_llm.py --weight full_sft --hidden_size 768 --num_hidden_layers 8 --device cuda --max_new_tokens 128
printf '0\n' | python eval_llm.py --weight small384l8_sft --hidden_size 384 --num_hidden_layers 8 --device cuda --max_new_tokens 128
printf '0\n' | python eval_llm.py --weight small384l8_distill --hidden_size 384 --num_hidden_layers 8 --device cuda --max_new_tokens 128
```

## Verification and remaining limits

Five regression tests passed. The small-model test checks exact parameter count,
finite forward/backward, tied embeddings, and strict save/reload equivalence.
A two-record distillation smoke run and resumed second epoch passed, reaching
epoch index 1, step 2. The three pilot checkpoints had finite tensors and expected
final resume positions: pretrain step 512, SFT step 256, distillation step 512.
Strict loading passed during student evaluation. The copied teacher's SHA-256
still matched the original after execution.

Run tests with `python -m unittest discover -s tests -v`. Local logs, exact
commands, hashes, metric JSON files, and chat outputs reside in `runs/smaller/`
and are excluded from Git along with datasets, weights, and internal plans.

Smaller-student RL, real reward-model integration, quantization, full-corpus
training, and long-run quality remain unverified. PPO/GRPO/agent reward-model
memory does not shrink just because the student is smaller. Future student RL
must use the matching width/layer flags and explicit student `--from_weight`
prefix; these pilot weights are not recommended for that next stage yet.
