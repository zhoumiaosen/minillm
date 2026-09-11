# Dataset inputs

No training datasets are bundled. The loaders consume JSONL: one JSON object
per line. Use the upstream `dataset/dataset.md` for dataset references. Obtain
data separately and review its terms before redistribution.

| Stage | Default file in `dataset/` | Required fields |
| --- | --- | --- |
| Pretrain | `pretrain_t2t_mini.jsonl` | `text` string |
| SFT | `sft_t2t_mini.jsonl` | `conversations` list |
| LoRA | `lora_medical.jsonl` | `conversations` list |
| Distillation | `sft_t2t_mini.jsonl` | `conversations` list; teacher weights also needed |
| PPO / GRPO | `rlaif.jsonl` | `conversations` list, ending in an assistant entry |
| DPO | `dpo.jsonl` | `chosen` and `rejected` conversation lists |
| Agent RL | `agent_rl.jsonl` | `conversations`, tool schemas, and `gt` list |

Pretraining line:

```json
{"text": "A short training passage."}
```

SFT or RLAIF line:

```json
{"conversations": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4"}]}
```

RLAIF uses all entries except the last as its generation prompt. Retain the
final assistant entry; a user-only conversation would lose its last question.

DPO line:

```json
{"chosen": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4"}], "rejected": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "5"}]}
```

Agent RL reads tool schemas from the system message's `tools` field (a JSON
string or list), removes the last conversation entry, and checks generated
answers against the `gt` list. Match tool names and argument schemas to `TOOLS`
in `trainer/train_agent.py`. The examples above describe formats, not a useful
training corpus. Keep complete datasets out of Git.
