#!/usr/bin/env python3
"""Items for the emotion-context experiment on IntentQA (test) and Social-IQ 2.0 (val).

    .venv/bin/python scripts/emoctx_manifests.py

Two files per dataset, both derived from the item set the transfer run fixed
(``results/newdata/intentqa_socialiq_20260923/manifests``), so the questions,
the videos and the exclusions are exactly the ones already reported:

``<dataset>_videos.jsonl``
    One harness Example per video, carrying the MME-Emotion ER_Wild prompt
    verbatim ("identify the most prominent emotion displayed in the video.
    Candidate labels: neutral, happy, angry, sad, fear, surprise, disgust").
    This is the harness's native task: no question reaches it, so the context
    it writes is question-independent and shared by every question on the clip.
    ``labels`` is a schema-required placeholder; these items have no gold
    emotion, the harness never reads gold labels, and nothing scores them.

``<dataset>_qa.jsonl``
    One multiple-choice question per row with the ORIGINAL question and option
    texts from the dataset files (the transfer manifests had to strip commas to
    fit the label grammar; a native lettered prompt does not). Every row is
    checked against the transfer manifest: same id, same video, same gold
    index, and the sanitised options must equal that manifest's label set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

from emoharness.data.manifest import Example, read_manifest
from emoharness.data.prompt_template import render_prompt_for_task
from emoharness.data.tasks import Split, Task

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results/newdata/intentqa_socialiq_20260923/manifests'
OUT = ROOT / 'results/newdata/emotion_context_20260925/manifests'
IQA_CSV = ROOT / 'data/raw/intentqa/test.csv'
SIQ_QA = ROOT / 'data/raw/social_iq_video/qa/qa_val.json'

#: One of the ER_Wild label sets the harness was built on (249 MME-Emotion items).
EMOTIONS = ('neutral', 'happy', 'angry', 'sad', 'fear', 'surprise', 'disgust')
PLACEHOLDER_GOLD = ('neutral',)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(option: str) -> str:
    """The transfer manifests' option rendering, used here only to cross-check."""
    text = option.replace(';', ' -').replace(',', '')
    return ' '.join(text.split())


def source_examples(dataset: str, record: dict) -> tuple[Example, ...]:
    declared = record[dataset]
    path = ROOT / declared['file']
    if sha256(path) != declared['sha256']:
        raise SystemExit(f'{path.name} no longer matches manifest_record.json')
    return read_manifest(path)


def raw_rows(dataset: str) -> dict[str, tuple[str, list[str], int]]:
    """example id -> (question, original options, gold index), ids as the transfer run built them."""
    rows: dict[str, tuple[str, list[str], int]] = {}
    if dataset == 'intentqa_test':
        for row in csv.DictReader(IQA_CSV.open(encoding='utf-8')):
            eid = f'intentqa_test_{row["video_id"]}_{row["qid"]}'
            rows[eid] = (
                row['question'],
                [row[f'a{i}'] for i in range(5)],
                int(row['answer']),
            )
    else:
        text = SIQ_QA.read_text(encoding='utf-8')
        try:
            data = json.loads(text)
            data = data if isinstance(data, list) else list(data.values())
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        for index, row in enumerate(data):
            eid = f'social_iq_val_{index:04d}_{row["qid"]}'
            rows[eid] = (
                row['q'],
                [row[f'a{i}'] for i in range(4)],
                int(row['answer_idx']),
            )
    return rows


def build(dataset: str, record: dict) -> dict:
    examples = source_examples(dataset, record)
    raw = raw_rows(dataset)
    qa, videos = [], {}
    for e in examples:
        question, options, gold = raw[e.example_id]
        if gold != e.metadata['answer_index'] or [sanitize(o) for o in options] != list(
            e.label_set
        ):
            raise SystemExit(
                f'{e.example_id}: raw row disagrees with the transfer manifest'
            )
        qa.append(
            {
                'example_id': e.example_id,
                'video_id': e.source_id,
                'question': ' '.join(question.split()),
                'options': [' '.join(o.split()) for o in options],
                'answer_index': gold,
                'media_path': e.media_path,
                'media_hash': e.media_hash,
            }
        )
        videos.setdefault(e.source_id, e)
    prompt = render_prompt_for_task(Task.ER_WILD, EMOTIONS)
    video_examples = [
        Example(
            example_id=f'{dataset}_video_{vid}',
            source_id=vid,
            origin=e.origin,
            task=None,
            kind=None,
            split=Split.EXT_TEST,
            media_path=e.media_path,
            media_hash=e.media_hash,
            prompt=prompt,
            label_set=EMOTIONS,
            labels=PLACEHOLDER_GOLD,
            metadata={'dataset': dataset, 'video_id': vid, 'gold_is_placeholder': True},
        )
        for vid, e in sorted(videos.items())
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, lines in (
        (f'{dataset}_videos.jsonl', [v.model_dump_json() for v in video_examples]),
        (f'{dataset}_qa.jsonl', [json.dumps(q, ensure_ascii=False) for q in qa]),
    ):
        path = OUT / name
        tmp = path.with_suffix('.jsonl.pending')
        tmp.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
        tmp.replace(path)
        files[name.split('_')[-1].split('.')[0]] = {
            'file': str(path.relative_to(ROOT)),
            'sha256': sha256(path),
            'n': len(lines),
        }
    return {
        **files,
        'n_videos': len(video_examples),
        'n_questions': len(qa),
        'emotion_prompt': prompt,
        'source_manifest': record[dataset]['file'],
        'source_manifest_sha256': record[dataset]['sha256'],
    }


def main() -> int:
    record = json.loads((SOURCE / 'manifest_record.json').read_text())
    out = {
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'emotion_labels': list(EMOTIONS),
    }
    for dataset in ('intentqa_test', 'social_iq_val'):
        out[dataset] = build(dataset, record)
    (OUT / 'manifest_record.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
