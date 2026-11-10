"""Convert official SQuAD-format files into complete, bounded SFT examples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def context_key(text):
    return hashlib.sha256(''.join(text.split()).encode()).hexdigest()


def make_example(context, qa, source):
    impossible = qa.get('is_impossible', False)
    answers = qa['answers']
    if not impossible:
        if not answers or any(not a['text'] or
                              context[a['answer_start']:a['answer_start'] + len(a['text'])] != a['text']
                              for a in answers):
            raise ValueError(f"Invalid answer span: {qa['id']}")
    chinese = source == 'cmrc'
    refusal = '资料中未提供。' if chinese else 'Not provided in the context.'
    instruction = ('仅根据资料回答问题。只输出简短答案；没有答案时输出“资料中未提供。”。'
                   if chinese else 'Answer using only the context. Give a short answer. '
                   'If the answer is absent, say "Not provided in the context."')
    prompt = (f"资料：{context}\n问题：{qa['question']}" if chinese
              else f"Context: {context}\nQuestion: {qa['question']}")
    return {'id': qa['id'], 'source': source, 'context_sha256': context_key(context),
            'unanswerable': impossible,
            'answers': [refusal] if impossible else [a['text'] for a in answers],
            'conversations': [{'role': 'system', 'content': instruction},
                              {'role': 'user', 'content': prompt},
                              {'role': 'assistant', 'content': refusal if impossible else answers[0]['text']}]}


def paragraphs(path):
    for article in json.loads(path.read_text())['data']:
        yield from article['paragraphs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('runs/context-qa'))
    args = parser.parse_args()
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(str(root / 'model'), local_files_only=True)
    folder = args.data_dir.resolve()
    raw = folder / 'raw'
    validation_contexts = {context_key(p['context']) for source in ('cmrc', 'squad')
                           for p in paragraphs(raw / f'{source}-dev.json')}
    counts = Counter()
    invalid_ids = []
    output = {}
    for source in ('cmrc', 'squad'):
        for split in ('train', 'dev'):
            print(f'Checking {source} {split}...', flush=True)
            rows = []
            seen = set()
            for p in paragraphs(raw / f'{source}-{split}.json'):
                for qa in p['qas']:
                    counts[f'{source}_{split}_raw'] += 1
                    if split == 'train' and context_key(p['context']) in validation_contexts:
                        counts[f'{source}_{split}_overlap_removed'] += 1
                        continue
                    try:
                        row = make_example(p['context'], qa, source)
                    except ValueError:
                        counts[f'{source}_{split}_invalid_span'] += 1
                        invalid_ids.append(f"{source}/{split}/{qa['id']}")
                        continue
                    signature = (row['context_sha256'], qa['question'])
                    if signature in seen:
                        counts[f'{source}_{split}_duplicates'] += 1
                        continue
                    seen.add(signature)
                    # Count the longest reference too, reserving space for generation in evaluation.
                    lengths = []
                    for answer in row['answers']:
                        messages = row['conversations'][:-1] + [{'role': 'assistant', 'content': answer}]
                        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
                        lengths.append(len(tokenizer(rendered).input_ids))
                    if max(lengths) > 512:
                        counts[f'{source}_{split}_too_long'] += 1
                        continue
                    rows.append(row)
            output[source, split] = rows
            counts[f'{source}_{split}_eligible'] = len(rows)
            print(f'{source} {split}: {len(rows)} complete examples fit 512 tokens.', flush=True)
    # Equal language counts; retain the eligible English answerability ratio.
    rng = random.Random(42)
    size = min(len(output[s, 'train']) for s in ('cmrc', 'squad'))
    train = []
    for source in ('cmrc', 'squad'):
        rows = output[source, 'train']
        rng.shuffle(rows)
        for row in rows[:size]:
            counts[f'{source}_train_selected'] += 1
            counts[f'{source}_train_unanswerable'] += int(row['unanswerable'])
        train.extend(rows[:size])
    rng.shuffle(train)
    assert size > 0
    assert not ({r['context_sha256'] for r in train} & validation_contexts)
    files = {'train.jsonl': train}
    files.update({f'{s}-validation.jsonl': output[s, 'dev'] for s in ('cmrc', 'squad')})
    for name, rows in files.items():
        with (folder / name).open('w') as f:
            for row in rows:
                # Training loader requires its existing conversations-only schema.
                item = {'conversations': row['conversations']} if name == 'train.jsonl' else row
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    manifest = {'max_tokens': 512, 'seed': 42, 'counts': dict(counts), 'invalid_ids': invalid_ids,
                'license': 'CC-BY-SA-4.0 for both datasets; retain original attribution',
                'sources': {'cmrc': 'https://github.com/ymcui/cmrc2018',
                            'squad': 'https://rajpurkar.github.io/SQuAD-explorer/'},
                'sha256': {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in list(raw.glob('*.json')) + [folder / n for n in files]}}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
