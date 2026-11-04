"""Items for the emotion-context experiment on IntentQA (test) and Social-IQ 2.0 (val).

    python -m experiments.emotion_context.manifests [--only <dataset>]

Two files per dataset, both derived from the item set the transfer run fixed
(``results/transfer/manifests``), so the questions,
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

``social_iq_emotion`` is a third item set: the Social-IQ 2.0 questions that
explicitly ask about emotion (the ``core`` rule of
``experiments.emotion_context.subset``, which reads only the question text).
Val has only 262 of them, so the train split is added: it carries answers, and
neither the harnesses nor the students were built on it. Val rows are copied
from ``social_iq_val_qa.jsonl`` unchanged; train rows pass the transfer
manifests' own inclusion checks (``experiments.transfer.manifests.render``),
so both splits are filtered by the same rules. Every row records its split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from emoharness.data.manifest import Example, read_manifest
from emoharness.data.prompt_template import render_prompt_for_task
from emoharness.data.tasks import Split, Task
from experiments.common import ROOT
from experiments.emotion_context.subset import subset_of
from experiments.transfer.manifests import SIQ_VIDEOS, render

SOURCE = ROOT / 'results/transfer/manifests'
OUT = ROOT / 'results/emotion_context/manifests'
IQA_CSV = ROOT / 'data/raw/intentqa/test.csv'
SIQ_QA = ROOT / 'data/raw/social_iq_video/qa/qa_val.json'
SIQ_TRAIN_QA = ROOT / 'data/raw/social_iq_video/qa/qa_train.json'
EMOTION_SET = 'social_iq_emotion'

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


def social_iq_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding='utf-8')
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else list(data.values())
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


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
        for index, row in enumerate(social_iq_rows(SIQ_QA)):
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
    video_examples = [
        video_example(f'{dataset}_video_{vid}', e, {'dataset': dataset})
        for vid, e in sorted(videos.items())
    ]
    return {
        **write(dataset, qa, video_examples),
        'source_manifest': record[dataset]['file'],
        'source_manifest_sha256': record[dataset]['sha256'],
    }


def video_example(example_id: str, clip: Example, metadata: dict) -> Example:
    """The harness's native item for one clip: the ER_Wild prompt, no question."""
    return Example(
        example_id=example_id,
        source_id=clip.source_id,
        origin=clip.origin,
        task=None,
        kind=None,
        split=Split.EXT_TEST,
        media_path=clip.media_path,
        media_hash=clip.media_hash,
        prompt=render_prompt_for_task(Task.ER_WILD, EMOTIONS),
        label_set=EMOTIONS,
        labels=PLACEHOLDER_GOLD,
        metadata={**metadata, 'video_id': clip.source_id, 'gold_is_placeholder': True},
    )


def write(dataset: str, qa: list[dict], video_examples: list[Example]) -> dict:
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
        'emotion_prompt': video_examples[0].prompt,
    }


def build_emotion_subset(record: dict) -> dict:
    """Social-IQ 2.0 core emotion questions, val (as already run) plus train."""
    declared = record['social_iq_val']['qa']
    path = ROOT / declared['file']
    if sha256(path) != declared['sha256']:
        raise SystemExit(f'{path.name} no longer matches manifest_record.json')
    qa = [
        {**q, 'split': 'val'}
        for q in map(json.loads, path.read_text(encoding='utf-8').splitlines())
        if subset_of(q['question']) == 'core'
    ]
    vpath = ROOT / record['social_iq_val']['videos']['file']
    clips = {e.source_id: ('val', e) for e in read_manifest(vpath)}
    dropped: dict[str, int] = {}
    media_hashes: dict = {}
    # The row index makes ids unique, as for val: qid repeats within a split.
    for index, row in enumerate(social_iq_rows(SIQ_TRAIN_QA)):
        question = ' '.join(row['q'].split())
        if subset_of(question) != 'core':
            continue
        eid = f'social_iq_train_{index:04d}_{row["qid"]}'
        options = [row[f'a{i}'] for i in range(4)]
        e, why = render(
            eid,
            row['vid_name'],
            'social_iq_2',
            row['q'],
            options,
            int(row['answer_idx']),
            SIQ_VIDEOS / f'{row["vid_name"]}.mp4',
            {},
            media_hashes,
        )
        if e is None:
            reason = str(why)
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        qa.append(
            {
                'example_id': eid,
                'video_id': e.source_id,
                'question': question,
                'options': [' '.join(o.split()) for o in options],
                'answer_index': int(row['answer_idx']),
                'media_path': e.media_path,
                'media_hash': e.media_hash,
                'split': 'train',
            }
        )
        clips.setdefault(e.source_id, ('train', e))
    used = {q['video_id'] for q in qa}
    video_examples = [
        video_example(
            f'social_iq_{split}_video_{vid}',
            e,
            {'dataset': EMOTION_SET, 'split': split},
        )
        for vid, (split, e) in sorted(clips.items())
        if vid in used
    ]
    return {
        **write(EMOTION_SET, qa, video_examples),
        'rule': 'experiments.emotion_context.subset.subset_of(question) == "core"',
        'by_split': {
            split: {
                'n_questions': sum(q['split'] == split for q in qa),
                'n_videos': sum(e.metadata['split'] == split for e in video_examples),
            }
            for split in ('val', 'train')
        },
        'train_excluded_by_reason': dropped,
        'sources': {
            'val': declared['file'],
            'val_sha256': declared['sha256'],
            'train': str(SIQ_TRAIN_QA.relative_to(ROOT)),
            'train_sha256': sha256(SIQ_TRAIN_QA),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--only',
        choices=('intentqa_test', 'social_iq_val', EMOTION_SET),
        help='rebuild one item set and keep the others as recorded',
    )
    args = parser.parse_args(argv)
    path = OUT / 'manifest_record.json'
    out = json.loads(path.read_text()) if args.only and path.is_file() else {}
    out.update(
        {
            'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'emotion_labels': list(EMOTIONS),
        }
    )
    record = json.loads((SOURCE / 'manifest_record.json').read_text())
    for dataset in ('intentqa_test', 'social_iq_val'):
        if args.only in (None, dataset):
            out[dataset] = build(dataset, record)
    if args.only in (None, EMOTION_SET):
        out[EMOTION_SET] = build_emotion_subset(out)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
