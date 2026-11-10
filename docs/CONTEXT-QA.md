# Context-based fine-tuning

This experiment continues the 384-wide, 8-layer, 17.2M full-SFT checkpoint.
It teaches short answers supported by a supplied passage, including abstention
when SQuAD labels the question unanswerable. It does not train a retriever.

## Data

- [CMRC 2018](https://github.com/ymcui/cmrc2018), Cui et al., EMNLP 2019:
  Simplified Chinese span extraction.
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), Rajpurkar et al.,
  ACL 2018: English span extraction and unanswerable questions.

Both datasets are distributed under CC BY-SA 4.0. Retain their attribution and
dataset license when distributing data derivatives; the repository code license
does not replace the dataset licenses. Downloaded data and generated JSONL remain
under ignored `runs/context-qa/` and are not included in Git commits.

Official source files, relative to `runs/context-qa/raw/`:

| Local file | Download |
| --- | --- |
| cmrc-train.json | https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_train.json |
| cmrc-dev.json | https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json |
| squad-train.json | https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json |
| squad-dev.json | https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json |

From the repository root, with the existing Python environment:

```bash
python scripts/prepare_context_qa.py
```

Preparation validates the annotated answer spans, excludes malformed annotations
(with their IDs and counts in the manifest), and uses the actual model
tokenizer/chat template. It skips examples exceeding 512 tokens, including the
complete answer. It does not crop around the known answer. Explicit system
instructions prevent the trainer from adding random generic system messages.

All whitespace-normalized validation passage hashes are excluded from training.
Duplicate passage/question pairs are removed within each split. Training contains
equal numbers of eligible CMRC and SQuAD examples, sampled with seed 42; SQuAD
sampling retains its unanswerable examples without forcing an exact ratio.
`manifest.json` records counts and SHA-256 hashes of raw and prepared files.

The prepared 2026-09-13 experiment contains 9,572 training examples: 4,786 from
each dataset, including 1,573 unanswerable SQuAD questions. Validation contains
1,392 CMRC and 10,812 SQuAD examples. Length filtering removed 5,335 CMRC training
examples and 9,994 SQuAD training examples. Invalid answer offsets removed 21
CMRC training and 176 CMRC development examples. No exact normalized validation
passage overlap was found in training. These counts describe this filtered run,
not the full datasets.

`cmrc-validation.jsonl` and `squad-validation.jsonl` retain every eligible official
development example with all reference answers and provenance. They are filtered
development subsets, not official full-benchmark scores. Original pretraining/SFT
overlap with these public datasets has not been ruled out. CMRC contains no
natural unanswerable training questions; Chinese abstention needs separate testing.

## Train

For this workstation, run:

```bash
PYTHON_BIN=/media/miaosen/AIs/workspace/MiniMind/minimind-clean/.venv/bin/python \
  bash /media/miaosen/AIs/workspace/MiniMind/minillm-small/scripts/finetune_context_qa.sh
```

Elsewhere, activate the project environment and run
`bash scripts/finetune_context_qa.sh` (uses `python3` by default).

The initial experiment is one epoch, batch 8, accumulation 1, sequence length 512,
learning rate 1e-5, BF16, seed 42. It saves every 250 batches and at completion.
Input: `out/small384l8_full_sft_384.pth`.
Output: `out/small384l8_context_qa_384.pth` and matching resume state in
`checkpoints/`. Logs: `runs/context-qa/train.log`.
The launcher refuses to overwrite a previous experiment. It runs in the foreground;
it does not schedule background training or automatic monitoring.

Before using the new checkpoint, compare it against the input checkpoint on both
validation subsets using the same system instructions and deterministic generation.
Measure answer exact match/token overlap and answerable versus unanswerable
performance separately. Also test fresh invented facts and general chat to detect
memorization and loss of other capabilities. Lower training loss alone is not
evidence that RAG is fixed. Do not add more epochs until these results are reviewed.
